#!/usr/bin/env bash
# Apache 2.0

# The launcher can be run from any directory: it finds its files from its own location
scripts_dir="$(dirname -- "$0")"

model="$1"
shift
python "$scripts_dir/task_1_finetuning_bert_cls.py" --config="$scripts_dir/../yaml/cls.yaml" --model="$model" "$@"
