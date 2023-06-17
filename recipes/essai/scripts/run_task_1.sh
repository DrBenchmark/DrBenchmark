#!/usr/bin/env bash
# Apache 2.0

python task_1_finetuning_bert_pos.py --config="../yaml/pos.yaml" --model="$1" --fewshot="$2"
