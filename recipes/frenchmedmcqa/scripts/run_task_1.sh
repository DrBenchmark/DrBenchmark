#!/usr/bin/env bash
# Apache 2.0

python task_1_finetuning_bert_mcqa.py --config="../yaml/mcqa.yaml" --model="$1" --fewshot="$2"
