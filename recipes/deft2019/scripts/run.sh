#!/usr/bin/env bash
# Apache 2.0

# The launcher can be run from any directory: it finds its files from its own location
scripts_dir="$(dirname -- "$0")"

model="$1"
shift
python "$scripts_dir/finetuning_bert_ner.py" --config="$scripts_dir/../yaml/ner.yaml" --model="$model" "$@"
