#!/usr/bin/env bash
# Apache 2.0

python task_3_finetuning_bert_ner.py --config="../yaml/ner_spec.yaml" --model="$1" --fewshot="$2"
