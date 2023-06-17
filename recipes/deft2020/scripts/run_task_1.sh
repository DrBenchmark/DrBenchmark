#!/usr/bin/env bash
# Apache 2.0

python task_1_finetuning_bert_regr.py --config="../yaml/regr.yaml" --model="$1" --fewshot="$2"
