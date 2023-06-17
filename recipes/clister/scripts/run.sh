#!/usr/bin/env bash
# Apache 2.0

python finetuning_bert_regr.py --config="../yaml/regr.yaml" --model="$1" --fewshot="$2"
