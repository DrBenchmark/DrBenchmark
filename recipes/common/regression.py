#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Fine-tuning for the semantic textual similarity tasks: a score between 0 and 5 for a pair of sentences."""

import logging

from scipy import stats
from sklearn.metrics import root_mean_squared_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from . import training


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = root_mean_squared_error(labels, predictions)
    return {"rmse": rmse}


def EDRM(ref, systm, debug=False):
    maxVal = 5
    dsum = 0

    for id in ref.keys():
        if id in systm and systm[id] >= 0 and systm[id] <= 5:
            d = abs(ref[id] - systm[id])
            if debug:
                logging.info(f"d: {d}")
            if abs(0 - systm[id]) > abs(maxVal - systm[id]):
                dmax = abs(0 - systm[id])
            else:
                dmax = abs(maxVal - systm[id])
        else:
            logging.info(f"{id} not in system answers!!!")
            d = maxVal
            dmax = maxVal
        if debug:
            logging.info(f"dmax: {dmax}")
        dsum += 1 - d / dmax
        if debug:
            logging.info(f"dsum: {dsum}")
    edrm = dsum / len(ref)
    return edrm


def SpMnCorr(ref, systm, alpha=0.05):
    r = [v for k, v in sorted(ref.items())]
    s = [v for k, v in sorted(systm.items())]

    if len(r) == len(s):
        c, p = stats.spearmanr(r, s)
        if p > alpha:
            logging.info("Spearman Correlation: reference and system result are NOT correlated")
        else:
            logging.info("Spearman Correlation: reference and system result are correlated")
        return [c, p]
    else:
        return ["error", "error"]


def scores(identifiers, references, predictions):
    """The scores of a run, from the similarity scores of the examples and their identifiers."""
    predictions = {id: p for id, p in zip(identifiers, predictions)}
    labels = {id: p for id, p in zip(identifiers, references)}

    edrm = EDRM(labels, predictions)
    logging.info(f">> EDRM: {edrm}")

    coeff, p = SpMnCorr(labels, predictions)
    logging.info(f">> Spearman Correlation: {coeff} ({p})")

    return {
        "EDRM": float(edrm),
        "spearman_correlation_coef": float(coeff),
        "spearman_correlation_p": float(p),
    }


class Task(training.Task):
    """A task that gives a similarity score to a pair of sentences.

    Args:
        dataset: name of the dataset on the HuggingFace Hub.
        text_columns: the two columns with the sentences to compare.
        label_column: column with the similarity score.
    """

    kind = "semantic textual similarity"
    model_class = AutoModelForSequenceClassification

    def __init__(self, dataset, text_columns, label_column):
        self.dataset = dataset
        self.text_columns = text_columns
        self.inputs = {column: "text" for column in text_columns}
        self.label_column = label_column

    def encode(self, example, tokenizer, max_length, model_name=None):
        first_column, second_column = self.text_columns
        text = f"{tokenizer.cls_token} {example[first_column]} {tokenizer.sep_token}  {example[second_column]} {tokenizer.eos_token}"
        return {"inputs": dict(tokenizer(text, truncation=True, max_length=max_length, padding="do_not_pad")), "text": text}

    def train(self):
        run = training.start(self.dataset)
        args, dataset = run.args, run.dataset
        model_name = training.model_of(args)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

        def preprocess_function(e):

            encoded = self.encode(e, tokenizer, args.max_position_embeddings, model_name)

            res = encoded["inputs"]
            res["text"] = encoded["text"]

            res["label"] = float(e[self.label_column])

            return res

        dataset_train, dataset_val, dataset_test = [
            split.remove_columns(["text"])
            for split in training.prepare_splits(run, preprocess_function, batched=False)
        ]
        for split in (dataset_train, dataset_val, dataset_test):
            split.set_format("torch")
        dataset_test_ids = list(dataset["test"]["id"])

        training_args = training.training_arguments(run, "rmse")

        trainer = training.create_trainer(model, training_args, dataset_train, dataset_val, tokenizer, compute_metrics)

        training.train(run, trainer)

        logging.info("***** Starting Evaluation *****")
        _predictions, _labels, _ = trainer.predict(dataset_test)
        _predictions = _predictions.reshape(-1)

        training.save_results(
            run,
            trainer,
            metrics=scores(dataset_test_ids, _labels, _predictions),
            identifiers=dataset_test_ids,
            real_labels=_labels.tolist(),
            system_predictions=[p for p in _predictions.tolist()],
        )
