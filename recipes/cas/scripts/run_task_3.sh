#!/usr/bin/env bash
# Apache 2.0

model="$1"
shift
python task_3_finetuning_bert_ner.py --config="../yaml/ner_neg.yaml" --model="$model" "$@"
