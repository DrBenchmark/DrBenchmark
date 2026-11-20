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


def scores(references, predictions):
    """The scores of a run, from the classes of the examples given by their numbers."""
    return classification_report(references, predictions, zero_division=.0, output_dict=True)


class Task(training.Task):
    """A task that gives one class to an example.

    Args:
        dataset: name of the dataset on the HuggingFace Hub.
        build_text: function `(example, tokenizer) -> str` that returns the text given to the model.
        inputs: the columns of an example that `build_text` reads: {column: "text" or "words"}.
        label_column: column with the class of the examples.
        id_column: column with the identifier of the examples.
    """

    kind = "classification"
    model_class = AutoModelForSequenceClassification

    def __init__(self, dataset, build_text, inputs, label_column="label", id_column="id"):
        self.dataset = dataset
        self.build_text = build_text
        self.inputs = inputs
        self.label_column = label_column
        self.id_column = id_column

    def label_names(self, dataset):
        """Names of the classes: the outputs of the model, in their order."""
        return dataset["train"].features[self.label_column].names

    def target(self, example, labels, features):
        """Number of the class of an example of the dataset: what the model learns."""
        return example[self.label_column]

    def encode(self, example, tokenizer, max_length, model_name=None):
        text = self.build_text(example, tokenizer)
        return {"inputs": dict(tokenizer(text, truncation=True, max_length=max_length, padding="do_not_pad")), "text": text}

    def results(self, references, predictions, labels):
        """What a run writes, from the classes of the test set given by their numbers: its scores, the references and the predictions."""
        logging.info(classification_report(
            references,
            predictions,
            digits=4,
            labels=range(len(labels)),
            target_names=labels,
            zero_division=.0
        ))
        return scores(references, predictions), references.tolist(), predictions.tolist()

    def train(self):
        run = training.start(self.dataset)
        args, dataset = run.args, run.dataset
        features = dataset["train"].features
        model_name = training.model_of(args)

        labels_list = self.label_names(dataset)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list))
        training.name_outputs(model, labels_list)

        def preprocess_function(e):

            encoded = self.encode(e, tokenizer, args.max_position_embeddings, model_name)

            res = encoded["inputs"]
            res["text"] = encoded["text"]

            res["label"] = self.target(e, labels_list, features)

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

        metrics, real_labels, system_predictions = self.results(labels, predictions, labels_list)

        training.save_results(
            run,
            trainer,
            metrics=metrics,
            identifiers=list(dataset["test"][self.id_column]),
            real_labels=real_labels,
            system_predictions=system_predictions,
        )
