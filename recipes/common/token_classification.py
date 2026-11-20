#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Fine-tuning for the token classification tasks: part-of-speech tagging and named-entity recognition."""

import logging

import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification

from . import training


def getConfig(raw_labels):

    label2id = {}
    id2label = {}

    for i, class_name in enumerate(raw_labels):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    return label2id, id2label


def main(dataset, tag_column="ner_tags", metric_for_best_model="f1", needs_data_dir=False):
    """Fine-tune and evaluate a model on a token classification task.

    Args:
        dataset: name of the dataset on the HuggingFace Hub, or path of its loading script.
        tag_column: column with the tags of the tokens.
        metric_for_best_model: metric computed on the validation set to choose the best
            checkpoint: "f1", "accuracy", "precision", "recall" or "macro_f1".
        needs_data_dir: give `--data_dir` to the loading script of the dataset.
    """
    run = training.start(dataset, needs_data_dir=needs_data_dir)
    args, dataset = run.args, run.dataset

    try:
        label_list = dataset["train"].features[tag_column].feature.names
    except AttributeError:
        # For datasets<4
        label_list = dataset["train"].features[tag_column][0].names

    label2id, id2label = getConfig(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
    model.config.label2id = label2id
    model.config.id2label = id2label

    def tokenize_and_align_labels(examples):

        label_all_tokens = False
        # label_all_tokens = True

        if args.model_name.lower().find("flaubert") != -1:
            if not args.max_position_embeddings:
                args.max_position_embeddings = tokenizer.model_max_length
            tokenized_inputs = []
            _labels = []

            # For sentence in batch
            for _e, _label in zip(examples["tokens"], examples[tag_column]):

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

                tokenized_inputs.append(_local)
                _labels.append(_local_labels)

            tokenized_inputs = {
                "input_ids": tokenized_inputs,
                "labels": _labels,
            }

        else:

            tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, max_length=args.max_position_embeddings, padding="do_not_pad", is_split_into_words=True)

            labels = []

            for i, label in enumerate(examples[tag_column]):

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

    train_tokenized_datasets, validation_tokenized_datasets, test_tokenized_datasets = training.prepare_splits(
        run, tokenize_and_align_labels, batched=True, keep_in_memory=True,
    )
    if "label" in dataset["train"].column_names:
        # A column of labels for another task (PxCorpus) would be taken for the tags of the tokens
        train_tokenized_datasets = train_tokenized_datasets.remove_columns(["label"])
        validation_tokenized_datasets = validation_tokenized_datasets.remove_columns(["label"])
        test_tokenized_datasets = test_tokenized_datasets.remove_columns(["label"])

    training_args = training.training_arguments(run, metric_for_best_model, greater_is_better=True)

    logging.info('Load Metrics')
    metric = evaluate.load("../../../metrics/seqeval.py", experiment_id=run.name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(p):
        # Beware : The labels here might be truncated according to `model_max_length`
        # These are the training metrics, not the final metrics.

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=.0)

        macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
        macro_f1 = sum(macro_values) / len(macro_values) if macro_values else 0.0

        return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

    trainer = training.create_trainer(
        model,
        training_args,
        train_tokenized_datasets,
        validation_tokenized_datasets,
        tokenizer,
        compute_metrics,
        data_collator=data_collator,
    )

    training.train(run, trainer)

    logging.info("***** Starting Evaluation *****")

    predictions, labels, _ = trainer.predict(test_tokenized_datasets)
    predictions = np.argmax(predictions, axis=2)

    # Convert predictions at the subtoken level to token level
    _true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # We might have truncated the labels in `tokenize_and_align_labels`
    # We retrieve the untruncated labels
    _true_labels = [[label_list[label] for label in doc] for doc in test_tokenized_datasets[tag_column]]
    # We pad the predictions with 'O' to match untruncated labels
    _true_predictions = [p + ['O'] * (len(l) - len(p)) for p, l in zip(_true_predictions, _true_labels)]

    cr_metric = metric.compute(predictions=_true_predictions, references=_true_labels, zero_division=.0)
    logging.info(cr_metric)

    training.save_results(
        run,
        trainer,
        metrics=cr_metric,
        identifiers=list(dataset["test"]["id"]),
        real_labels=_true_labels,
        system_predictions=_true_predictions,
    )
