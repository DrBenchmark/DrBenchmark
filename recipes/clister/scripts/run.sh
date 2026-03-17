#!/usr/bin/env bash
# Apache 2.0

model="$1"
shift
python finetuning_bert_regr.py --config="../yaml/regr.yaml" --model="$model" "$@"
