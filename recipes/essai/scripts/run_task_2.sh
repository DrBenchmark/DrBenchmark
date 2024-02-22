#!/usr/bin/env bash
# Apache 2.0

python task_2_finetuning_bert_cls.py --config="../yaml/cls.yaml" --model="$1" --fewshot="$2"
