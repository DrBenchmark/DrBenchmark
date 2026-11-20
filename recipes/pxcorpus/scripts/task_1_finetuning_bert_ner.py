#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 12/06/2023 Yanis LABRAK
# Apache 2.0

import sys
from pathlib import Path

# The code shared by the recipes is in `recipes/common/`: its parent is added to the search path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import token_classification  # noqa: E402

if __name__ == '__main__':
    token_classification.main(dataset="DrBenchmark/PxCorpus", metric_for_best_model="f1")
