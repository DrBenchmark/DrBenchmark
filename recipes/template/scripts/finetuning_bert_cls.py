#!/usr/bin/env python3

# Apache 2.0

"""Template of the script of a task: here, the classification of a text.

Copy the directory of this recipe and change what is marked TODO: see recipes/ADDING_A_TASK.md.
"""

import sys
from pathlib import Path

# The code shared by the recipes is in `recipes/common/`: its parent is added to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import utils  # noqa: E402

if __name__ == '__main__':
    # A run whose results exist stops here, before the libraries that train are loaded: it takes seconds
    utils.stop_if_already_computed()

from common import sequence_classification  # noqa: E402


def build_text(e, tokenizer):
    # TODO: the text that the model reads for an example of the dataset
    return e["text"]


# What is specific to the task: the code that the recipes share does the rest
# TODO: the dataset, the columns of an example that `build_text` reads, and the column of its class
TASK = sequence_classification.Task(dataset="DrBenchmark/MyCorpus", build_text=build_text, inputs={"text": "text"}, label_column="label")

if __name__ == '__main__':
    TASK.train()
