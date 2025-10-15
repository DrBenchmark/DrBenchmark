#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     To validate 01/05/2023 Yanis LABRAK
# Apache 2.0

import os
import json
import uuid
import shutil
import logging
import dataclasses

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from utils import parse_args


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=.0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


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
            "DrBenchmark/CAS",
            name=str(args.subset),
            trust_remote_code=True,
        )

    labels_list = dataset["train"].features["label"].names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list))

    def preprocess_function(e):

        text = f"{' '.join(e['tokens'])}"

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="max_length")
        res["text"] = text

        res["label"] = e["label"]

        return res

    dataset_train = dataset["train"].map(preprocess_function, batched=False).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(range(int(len(dataset_train) * args.fewshot)))
    if args.max_train_samples:
        dataset_train = dataset_train.select(range(args.max_train_samples))
    dataset_train = dataset_train.remove_columns(["text"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    if args.max_val_samples:
        dataset_val = dataset_val.select(range(args.max_val_samples))
    dataset_val = dataset_val.remove_columns(["text"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    if args.max_test_samples:
        dataset_test = dataset_test.select(range(args.max_test_samples))
    dataset_test = dataset_test.remove_columns(["text"])
    dataset_test.set_format("torch")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-CAS-{args.subset}-{uuid.uuid4().hex}"

    training_args = TrainingArguments(
        f"{args.output_dir}/{output_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        num_train_epochs=int(args.epochs),
        weight_decay=float(args.weight_decay),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        save_only_model=True,
        save_total_limit=1,
        report_to='none',
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
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
    predictions, labels, _ = trainer.predict(dataset_test)
    predictions = np.argmax(predictions, axis=1)

    cr_metrics = classification_report(
        labels,
        predictions,
        digits=4,
        labels=range(len(labels_list)),
        target_names=labels_list,
        zero_division=.0
    )
    logging.info(cr_metrics)

    with open(f"../runs/{output_name}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": f"{args.output_dir}/{output_name}_best_model",
            "metrics": classification_report(labels, predictions, zero_division=.0, output_dict=True),
            "hyperparameters": vars(args),
            "predictions": {
                "identifiers": dataset["test"]["id"],
                "real_labels": labels.tolist(),
                "system_predictions": predictions.tolist(),
            },
            'trainer_state': dataclasses.asdict(trainer.state),
        }, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
