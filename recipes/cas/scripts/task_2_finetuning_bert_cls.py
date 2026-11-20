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
    return f"{' '.join(e['tokens'])}"


# What is specific to the task: the code that the recipes share does the rest
TASK = sequence_classification.Task(dataset="DrBenchmark/CAS", build_text=build_text, inputs={"tokens": "words"})


if __name__ == '__main__':
    TASK.train()
