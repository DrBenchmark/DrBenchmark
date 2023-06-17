#!/usr/bin/env bash
# Apache 2.0

python task_1_finetuning_bert_ner.py --config="../yaml/ner.yaml" --model="$1" --fewshot="$2"
