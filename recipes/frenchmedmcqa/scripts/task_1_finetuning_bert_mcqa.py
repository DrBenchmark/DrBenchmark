#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

import logging
import sys
from pathlib import Path

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# The code shared by the recipes is in `recipes/common/`: its parent is added to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import training  # noqa: E402
from common.sequence_classification import compute_metrics  # noqa: E402


def main():

    run = training.start("DrBenchmark/FrenchMedMCQA")
    args, dataset = run.args, run.dataset

    labels_list = ["c", "a", "e", "d", "b", "be", "ae", "bc", "bd", "ab", "de", "cd", "ac", "ad", "ce", "bce", "abc", "cde", "bcd", "ace", "ade", "abe", "acd", "bde", "abd", "abde", "abcd", "bcde", "abce", "acde", "abcde"]
    original_labels_list = dataset["train"].features["correct_answers"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list))

    def preprocess_function(e):

        concatenated_choices = f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in ["a", "b", "c", "d", "e"]])
        text = f"{tokenizer.cls_token} {e['question']} {tokenizer.sep_token} {concatenated_choices} {tokenizer.eos_token}"

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="do_not_pad")
        res["text"] = text

        labels_letters = [original_labels_list[l] for l in e["correct_answers"]]  # Get from the answers indexes to the original letters
        labels_letters = "".join(sorted(labels_letters))  # Transform the multi-label answers into a multi-class problem
        labels_letters = labels_list.index(labels_letters)  # Get the corresponding index of the class
        res["label"] = labels_letters

        return res

    dataset_train, dataset_val, dataset_test = training.prepare_splits(run, preprocess_function, batched=False)
    dataset_train = dataset_train.remove_columns(["text"])
    dataset_train.set_format("torch")
    dataset_val = dataset_val.remove_columns(["text"])
    dataset_val.set_format("torch")
    # The text of the test examples is kept: it is given to the pipeline below

    training_args = training.training_arguments(run, "accuracy")

    trainer = training.create_trainer(model, training_args, dataset_train, dataset_val, tokenizer, compute_metrics)

    training.train(run, trainer)

    logging.info("***** Starting Evaluation *****")
    pipe = pipeline(task="text-classification", model=run.model_path + "_best_model", top_k=None, device=training_args.device)

    def compute_accuracy_exact_match(preds, refs):
        exact_score = []
        for p, r in zip(preds, refs):
            exact_score.append(sorted(p) == sorted(r))
        return sum(exact_score) / len(exact_score)

    def compute_accuracy_hamming(preds, refs):
        corrects = [True for p in preds if p in refs]
        corrects = sum(corrects)
        total_refs = len(list(set(preds + refs)))
        return corrects / total_refs

    y_pred = []
    y_true = []

    hamming_scores = []

    for e in dataset_test:

        res = pipe(e["text"], truncation=True, max_length=model.config.max_position_embeddings)

        pred = int(res[0][0]["label"].split("_")[-1])
        pred = labels_list[pred]
        y_pred.append(pred)
        splitted_pred = sorted(list(pred))

        # Reference
        true_label = labels_list[e["label"]]
        y_true.append(true_label)
        splitted_true_label = sorted(list(true_label))

        # Compute hamming score
        score = compute_accuracy_hamming(splitted_pred, splitted_true_label)
        hamming_scores.append(score)

    logging.info(">> Hamming Score")
    hamming_score = sum(hamming_scores) / len(hamming_scores)
    logging.info(hamming_score)

    logging.info(">> Exact Match Ratio (EMR)")
    exact_match = compute_accuracy_exact_match(y_true, y_pred)
    logging.info(exact_match)

    training.save_results(
        run,
        trainer,
        metrics={
            "hamming_score": float(hamming_score),
            "exact_match": float(exact_match),
        },
        identifiers=list(dataset["test"]["id"]),
        real_labels=y_true,
        system_predictions=y_pred,
    )


if __name__ == '__main__':
    main()
