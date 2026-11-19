#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 28/04/2023 Yanis LABRAK
# Apache 2.0

from common import sequence_classification


def build_text(e, tokenizer):
    return f"{tokenizer.cls_token} {e['source']} {tokenizer.sep_token} (1) {e['cible_1']} {tokenizer.sep_token} (2) {e['cible_2']} {tokenizer.sep_token} (3) {e['cible_3']} {tokenizer.eos_token}"


if __name__ == '__main__':
    sequence_classification.main(dataset="DrBenchmark/DEFT2020", build_text=build_text, label_column="correct_cible")
