#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 28/04/2023 Yanis LABRAK
# Apache 2.0

import os
import json
import uuid
import shutil
import logging

import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils import parse_args


def getConfig(raw_labels):

    label2id = {}
    id2label = {}

    for i, class_name in enumerate(raw_labels):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    return label2id, id2label


def main():

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if args.offline:
        dataset = load_from_disk(f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/")
    else:
        dataset = load_dataset(
            "DrBenchmark/E3C",
            name=args.subset,
            trust_remote_code=True,
        )

    train_dataset = dataset["train"]
    dev_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    label_list = train_dataset.features[f"ner_tags"].feature.names

    label2id, id2label = getConfig(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
    model.config.label2id = label2id
    model.config.id2label = id2label

    def tokenize_and_align_labels(examples):

        label_all_tokens = False
        # label_all_tokens = True

        if args.model_name.lower().find("flaubert") != -1:

            tokenized_inputs = []
            _labels = []

            # For sentence in batch
            for _e, _label in zip(examples["tokens"], examples["ner_tags"]):

                _local = [tokenizer("<s>")["input_ids"][1]]
                _local_labels = [-100]

                # For token in sentence
                for _i, (_t, _lb) in enumerate(zip(_e, _label)):
                    tokens_word = tokenizer(_t)["input_ids"][1:-1]
                    _local.extend(tokens_word)
                    _local_labels.extend([_lb] * len(tokens_word))

                if len(_local) > 250:
                    logging.info(f">> {len(_local)}")

                _local = _local[0:args.max_position_embeddings - 1]
                _local_labels = _local_labels[0:args.max_position_embeddings - 1]

                _local.append(tokenizer("</s>")["input_ids"][1])
                _local_labels.append(-100)

                padding_left = args.max_position_embeddings - len(_local)
                if padding_left > 0:
                    _local.extend([tokenizer("<pad>")["input_ids"][1]] * padding_left)
                    _local_labels.extend([-100] * padding_left)

                tokenized_inputs.append(_local)
                _labels.append(_local_labels)

            tokenized_inputs = {
                "input_ids": tokenized_inputs,
                "labels": _labels,
            }

        else:

            tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, max_length=args.max_position_embeddings, padding='max_length', is_split_into_words=True)

            labels = []

            for i, label in enumerate(examples[f"ner_tags"]):

                label_ids = []
                previous_word_idx = None

                word_ids = tokenized_inputs.word_ids(batch_index=i)

                for word_idx in word_ids:

                    if word_idx is None:
                        label_ids.append(-100)

                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])

                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)

                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels

        return tokenized_inputs

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        train_tokenized_datasets = train_tokenized_datasets.select(range(int(len(train_tokenized_datasets) * args.fewshot)))
    if args.max_train_samples:
        train_tokenized_datasets = train_tokenized_datasets.select(range(args.max_train_samples))
    dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
    if args.max_val_samples:
        dev_tokenized_datasets = dev_tokenized_datasets.select(range(args.max_val_samples))
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
    if args.max_test_samples:
        test_tokenized_datasets = test_tokenized_datasets.select(range(args.max_test_samples))

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-E3C-ner-{args.subset}-{str(uuid.uuid4().hex)}"

    training_args = TrainingArguments(
        f"{args.output_dir}/{output_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        num_train_epochs=int(args.epochs),
        weight_decay=float(args.weight_decay),
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_only_model=True,
        save_total_limit=1,
        report_to='none',
    )

    logging.info('Load Metrics')
    metric = evaluate.load("../../../metrics/seqeval.py", experiment_id=output_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(p):

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=.0)

        macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
        macro_f1 = sum(macro_values) / len(macro_values)

        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=dev_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("***** Starting Training *****")
    trainer.train()
    trainer.evaluate()

    logging.info("***** Save the best model *****")
    trainer.save_model(f"{args.output_dir}/{output_name}_best_model")
    shutil.rmtree(f"{args.output_dir}/{output_name}")

    logging.info("***** Starting Evaluation *****")

    predictions, labels, _ = trainer.predict(test_tokenized_datasets)
    predictions = np.argmax(predictions, axis=2)

    _true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    _true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    cr_metric = metric.compute(predictions=_true_predictions, references=_true_labels, zero_division=.0)
    logging.info(cr_metric)

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open(f"../runs/{output_name}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": f"{args.output_dir}/{output_name}_best_model",
            "metrics": cr_metric,
            "hyperparameters": vars(args),
            "predictions": {
                "identifiers": dataset["test"]["id"],
                "real_labels": _true_labels,
                "system_predictions": _true_predictions,
            },
        }, f, ensure_ascii=False, indent=4, default=np_encoder)


if __name__ == '__main__':
    main()
