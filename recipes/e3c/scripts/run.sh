#!/usr/bin/env bash
# Apache 2.0

python finetuning_bert_ner.py --config="../yaml/ner.yaml" --model="$1" --subset="$2" --fewshot="$3"
