#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 29/04/2023 Yanis LABRAK
# Apache 2.0

from common import regression

if __name__ == '__main__':
    regression.main(dataset="DrBenchmark/DEFT2020", text_columns=("source", "cible"), label_column="moy")
