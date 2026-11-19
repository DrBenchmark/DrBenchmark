#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Fine-tuning for the multi-class classification tasks: one label per example."""

import logging

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from . import training


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


def main(dataset, build_text, label_column="label", id_column="id"):
    """Fine-tune and evaluate a model on a multi-class classification task.

    Args:
        dataset: name of the dataset on the HuggingFace Hub.
        build_text: function `(example, tokenizer) -> str` that returns the text given to the model.
        label_column: column with the class of the examples.
        id_column: column with the identifier of the examples.
    """
    run = training.start(dataset)
    args, dataset = run.args, run.dataset

    labels_list = dataset["train"].features[label_column].names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list))

    def preprocess_function(e):

        text = build_text(e, tokenizer)

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="do_not_pad")
        res["text"] = text

        res["label"] = e[label_column]

        return res

    dataset_train, dataset_val, dataset_test = [
        split.remove_columns(["text"])
        for split in training.prepare_splits(run, preprocess_function, batched=False)
    ]
    for split in (dataset_train, dataset_val, dataset_test):
        split.set_format("torch")

    training_args = training.training_arguments(run, "accuracy")

    trainer = training.create_trainer(model, training_args, dataset_train, dataset_val, tokenizer, compute_metrics)

    training.train(run, trainer)

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

    training.save_results(
        run,
        trainer,
        metrics=classification_report(labels, predictions, zero_division=.0, output_dict=True),
        identifiers=list(dataset["test"][id_column]),
        real_labels=labels.tolist(),
        system_predictions=predictions.tolist(),
    )
