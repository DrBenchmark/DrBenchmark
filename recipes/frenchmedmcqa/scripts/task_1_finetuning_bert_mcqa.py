#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

import logging
import sys
from pathlib import Path

# The code shared by the recipes is in `recipes/common/`: its parent is added to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import sequence_classification  # noqa: E402

LETTERS = ["a", "b", "c", "d", "e"]

# A question and its five answers
INPUTS = {"question": "text", **{f"answer_{letter}": "text" for letter in LETTERS}}

# The classes of the model: every set of right answers
LABELS = ["c", "a", "e", "d", "b", "be", "ae", "bc", "bd", "ab", "de", "cd", "ac", "ad", "ce", "bce", "abc", "cde", "bcd", "ace", "ade", "abe", "acd", "bde", "abd", "abde", "abcd", "bcde", "abce", "acde", "abcde"]


def build_text(e, tokenizer):
    concatenated_choices = f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in LETTERS])
    return f"{tokenizer.cls_token} {e['question']} {tokenizer.sep_token} {concatenated_choices} {tokenizer.eos_token}"


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


class MultipleChoiceTask(sequence_classification.Task):
    """A question with several right answers among five: the model chooses among the sets of answers.

    A prediction is the letters of the answers that the model takes for right, e.g. `ab`.
    """

    kind = "multiple-choice question answering"
    labels = LABELS

    def label_names(self, dataset):
        return self.labels

    def reference(self, example, labels, features):
        # Get from the answers indexes to the original letters
        names = features[self.label_column].feature.names
        # Transform the multi-label answers into a multi-class problem
        return "".join(sorted(names[answer] for answer in example[self.label_column]))

    def target(self, example, labels, features):
        # Get the corresponding index of the class
        return labels.index(self.reference(example, labels, features))

    def metrics(self, references, predictions, labels, identifiers=None):
        hamming_scores = [
            compute_accuracy_hamming(sorted(list(prediction)), sorted(list(reference)))
            for reference, prediction in zip(references, predictions)
        ]

        logging.info(">> Hamming Score")
        hamming_score = sum(hamming_scores) / len(hamming_scores)
        logging.info(hamming_score)

        logging.info(">> Exact Match Ratio (EMR)")
        exact_match = compute_accuracy_exact_match(references, predictions)
        logging.info(exact_match)

        return {
            "hamming_score": float(hamming_score),
            "exact_match": float(exact_match),
        }

    def results(self, references, predictions, labels):
        y_true = [labels[index] for index in references]
        y_pred = [labels[index] for index in predictions]
        return self.metrics(y_true, y_pred, labels), y_true, y_pred


# What is specific to the task: the code that the recipes share does the rest
TASK = MultipleChoiceTask(dataset="DrBenchmark/FrenchMedMCQA", build_text=build_text, inputs=INPUTS, label_column="correct_answers")


if __name__ == '__main__':
    TASK.train()
