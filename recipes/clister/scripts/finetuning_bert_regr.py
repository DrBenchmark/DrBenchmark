#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 29/04/2023 Yanis LABRAK
# Apache 2.0

from common import regression

if __name__ == '__main__':
    regression.main(dataset="DrBenchmark/CLISTER", text_columns=("text_1", "text_2"), label_column="label")
