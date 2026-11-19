#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 15/06/2023 Yanis LABRAK
# Apache 2.0

from common import token_classification

if __name__ == '__main__':
    token_classification.main(dataset="../../../data_loaders_hf/DEFT2019.py", metric_for_best_model="accuracy", needs_data_dir=True)
