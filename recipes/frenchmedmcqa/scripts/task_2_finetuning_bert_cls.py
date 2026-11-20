#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     To validate 01/05/2023 Yanis LABRAK
# Apache 2.0

import sys
from pathlib import Path

# The code shared by the recipes is in `recipes/common/`: its parent is added to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import sequence_classification  # noqa: E402


def build_text(e, tokenizer):
    concatenated_choices = f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in ["a", "b", "c", "d", "e"]])
    return f"{tokenizer.cls_token} {e['question']} {tokenizer.sep_token} {concatenated_choices} {tokenizer.eos_token}"


if __name__ == '__main__':
    sequence_classification.main(dataset="DrBenchmark/FrenchMedMCQA", build_text=build_text, label_column="number_correct_answers")
