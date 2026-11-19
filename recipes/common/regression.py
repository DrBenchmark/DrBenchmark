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
    return(edrm)


def SpMnCorr(ref, systm, alpha=0.05):
    r = [v for k, v in sorted(ref.items())]
    s = [v for k, v in sorted(systm.items())]

    if len(r) == len(s):
        c, p = stats.spearmanr(r, s)
        if p > alpha:
            logging.info("Spearman Correlation: reference and system result are NOT correlated")
        else:
            logging.info("Spearman Correlation: reference and system result are correlated")
        return([c, p])
    else:
        return(["error", "error"])


def main(dataset, text_columns, label_column):
    """Fine-tune and evaluate a model on a semantic textual similarity task.

    Args:
        dataset: name of the dataset on the HuggingFace Hub.
        text_columns: the two columns with the sentences to compare.
        label_column: column with the similarity score.
    """
    run = training.start(dataset)
    args, dataset = run.args, run.dataset
    first_column, second_column = text_columns

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    def preprocess_function(e):

        text = f"{tokenizer.cls_token} {e[first_column]} {tokenizer.sep_token}  {e[second_column]} {tokenizer.eos_token}"

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="do_not_pad")
        res["text"] = text

        res["label"] = float(e[label_column])

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
    predictions = {id: p for id, p in zip(dataset_test_ids, _predictions)}
    labels = {id: p for id, p in zip(dataset_test_ids, _labels)}

    edrm = EDRM(labels, predictions)
    logging.info(f">> EDRM: {edrm}")

    coeff, p = SpMnCorr(labels, predictions)
    logging.info(f">> Spearman Correlation: {coeff} ({p})")

    training.save_results(
        run,
        trainer,
        metrics={
            "EDRM": float(edrm),
            "spearman_correlation_coef": float(coeff),
            "spearman_correlation_p": float(p),
        },
        identifiers=dataset_test_ids,
        real_labels=_labels.tolist(),
        system_predictions=[p for p in _predictions.tolist()],
    )
