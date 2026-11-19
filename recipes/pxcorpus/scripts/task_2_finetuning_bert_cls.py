#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     To validate 12/06/2023 Yanis LABRAK
# Apache 2.0

from common import sequence_classification


def build_text(e, tokenizer):
    return f"{' '.join(e['tokens'])}"


if __name__ == '__main__':
    sequence_classification.main(dataset="DrBenchmark/PxCorpus", build_text=build_text)
