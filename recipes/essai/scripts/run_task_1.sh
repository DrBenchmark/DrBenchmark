#!/usr/bin/env bash
# Apache 2.0

model="$1"
shift
python task_1_finetuning_bert_pos.py --config="../yaml/pos.yaml" --model="$model" "$@"
