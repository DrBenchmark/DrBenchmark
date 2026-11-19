#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 15/06/2023 Yanis LABRAK
# Apache 2.0

from common import multi_label_classification

if __name__ == '__main__':
    multi_label_classification.main(dataset="DrBenchmark/DEFT2021", text_column="text")
