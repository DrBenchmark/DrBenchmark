#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Fine-tuning for the multi-label classification tasks: several labels per example.

The datasets give the labels of an example in the column `specialities`, and as a vector
of 0 and 1 in the column `specialities_one_hot`.
"""

import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

from . import training

THRESHOLD_VALUE = 0.70


def toLogits(predictions, threshold=THRESHOLD_VALUE):

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    return y_pred


def multi_label_metrics(predictions, labels, threshold=THRESHOLD_VALUE):

    y_pred = toLogits(predictions, threshold)
    y_true = labels

    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=.0)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=.0)
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=.0)
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {'f1_macro': f1_macro_average, 'f1_micro': f1_micro_average, 'f1_weighted': f1_weighted_average, 'accuracy': accuracy, 'roc': roc_auc}

    return metrics


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )
    return result


def scores(references, predictions):
    """The scores of a run, from the labels of the examples given as vectors of 0 and 1."""
    return classification_report(references, predictions, zero_division=.0, output_dict=True)


class Task(training.Task):
    """A task that gives several labels to a text.

    Args:
        dataset: name of the dataset on the HuggingFace Hub.
        text_column: column with the text given to the model.
    """

    kind = "multi-label classification"
    model_class = AutoModelForSequenceClassification

    def __init__(self, dataset, text_column):
        self.dataset = dataset
        self.text_column = text_column
        self.inputs = {text_column: "text"}

    def encode(self, example, tokenizer, max_length, model_name=None):
        return {"inputs": dict(tokenizer(example[self.text_column], truncation=True, max_length=max_length, padding="do_not_pad"))}

    def decode(self, logits, encoded, labels):
        """The labels whose probability reaches the threshold."""
        return [label for label, predicted in zip(labels, toLogits(logits[None], THRESHOLD_VALUE)[0]) if predicted]

    def train(self):
        text_column = self.text_column
        run = training.start(self.dataset, task=self)
        args, dataset = run.args, run.dataset
        model_name = training.model_of(args)

        labels_list = dataset["train"].features["specialities"].feature.names
        run.labels = labels_list

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list), problem_type="multi_label_classification")
        training.name_outputs(model, labels_list)

        def preprocess_function(e):
            res = self.encode(e, tokenizer, args.max_position_embeddings, model_name)["inputs"]
            res["labels"] = e["specialities_one_hot"]
            return res

        dataset_train, dataset_val, dataset_test = [
            split.remove_columns([text_column, "id", "specialities"])
            for split in training.prepare_splits(run, preprocess_function, batched=False)
        ]
        for split in (dataset_train, dataset_val, dataset_test):
            split.set_format("torch")

        training_args = training.training_arguments(run, "f1_weighted")

        trainer = training.create_trainer(model, training_args, dataset_train, dataset_val, tokenizer, compute_metrics)

        training.train(run, trainer)

        logging.info("***** Starting Evaluation *****")

        predictions, labels, _ = trainer.predict(dataset_test)

        predictions = toLogits(predictions, THRESHOLD_VALUE)

        metrics = multi_label_metrics(predictions, labels, THRESHOLD_VALUE)
        logging.info(metrics)

        cr = classification_report(labels, predictions, labels=range(len(labels_list)), target_names=labels_list, digits=4, zero_division=.0)
        logging.info(cr)

        training.save_results(
            run,
            trainer,
            metrics=scores(labels, predictions),
            identifiers=list(dataset["test"]["id"]),
            real_labels=labels.tolist(),
            system_predictions=predictions.tolist(),
        )
