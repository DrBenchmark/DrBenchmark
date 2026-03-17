#!/usr/bin/env bash
# Apache 2.0

model="$1"
shift
python task_1_finetuning_bert_cls.py --config="../yaml/cls.yaml" --model="$model" "$@"
