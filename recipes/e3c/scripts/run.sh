#!/usr/bin/env bash
# Apache 2.0

model="$1"
subset="$2"
shift 2
python finetuning_bert_ner.py --config="../yaml/ner.yaml" --model="$model" --subset="$subset" "$@"
