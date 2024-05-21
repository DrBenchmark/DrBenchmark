#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 29/04/2023 Yanis LABRAK
# Apache 2.0

import os
import shutil

import uuid
import json
import argparse
import logging

from utils import parse_args, TrainingArgumentsWithMPSSupport

import torch
import numpy as np
from scipy import stats
from datasets import load_dataset, load_from_disk

from sklearn.metrics import mean_squared_error

from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, Trainer, TrainingArguments, TextClassificationPipeline

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def EDRM(ref, systm, minVal=0, maxVal=5, boundAdjustment=True):
    n = len(ref)
    total_distance = 0

    for id in ref.keys():
        r = ref[id]

        if id not in systm:
            raise ValueError(f"System did not provide a score for the id {id}")
        elif boundAdjustment:
            # As the task delimits a range score from 0 to 5, every system should bound its output to this range
            h = max(min(systm[id], maxVal), minVal)
        else:
            h = systm[id]

        max_distance = max(abs(minVal - r), abs(maxVal - r))
        total_distance += 1 - abs(r - h) / max_distance

    return total_distance / n

def SpMnCorr(ref, systm, alpha=0.05):
    r = [v for k, v in sorted(ref.items())]
    s = [v for k, v in sorted(systm.items())]

    if len(r) == len(s):
        c,p = stats.spearmanr(r,s)
        if p > alpha:
            print("Spearman Correlation: reference and system result are not correlated")
        else:
            print("Spearman Correlation: reference and system result are correlated")
        return([c,p])
    else:
        return(["error","error"])

def main():

    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    #logger.setLevel(logging.INFO)

    if args.offline == True:   
        dataset = load_from_disk(f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/")
    else:            
        dataset = load_dataset(
            "Dr-BERT/CLISTER",
            name="source",
            data_dir=args.data_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    def preprocess_function(e):

        text = f"{tokenizer.cls_token} {e['text_1']} {tokenizer.sep_token}  {e['text_2']} {tokenizer.eos_token}"

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="max_length")
        res["text"] = text

        res["label"] = float(e["label"])

        return res

    dataset_train = dataset["train"].map(preprocess_function, batched=False).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(range(int(len(dataset_train) * args.fewshot)))
    dataset_train = dataset_train.remove_columns(["text"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    dataset_val = dataset_val.remove_columns(["text"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    dataset_test_ids = list(dataset["test"]["id"])
    dataset_test = dataset_test.remove_columns(["text"])
    dataset_test.set_format("torch")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-CLISTER-regression-{str(uuid.uuid4().hex)}"

    training_args = TrainingArguments(
        f"{args.output_dir}/{output_name}",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        num_train_epochs=int(args.epochs),
        weight_decay=float(args.weight_decay),
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("***** Starting Training *****")
    trainer.train()
    trainer.evaluate()

    logging.info("***** Save the best model *****")
    trainer.save_model(f"{args.output_dir}/{output_name}_best_model")
    shutil.rmtree(f"{args.output_dir}/{output_name}")

    logging.info("***** Starting Evaluation *****")
    _predictions, _labels, _ = trainer.predict(dataset_test)
    predictions = {id: p for id, p in zip(dataset_test_ids, _predictions)}
    labels      = {id: p for id, p in zip(dataset_test_ids, _labels)}
    
    edrm = EDRM(labels, predictions)
    print(">> EDRM: ", edrm)

    coeff, p = SpMnCorr(labels, predictions)
    print(">> Spearman Correlation: ", coeff, "(",p,")")

    with open(f"../runs/{output_name}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": f"{args.output_dir}/{output_name}_best_model",
            "metrics": {
                "EDRM": float(edrm),
                "spearman_correlation_coef": float(coeff),
                "spearman_correlation_p": float(p),
            },
            "hyperparameters": vars(args),
            "predictions": {
                "identifiers": dataset_test_ids,
                "real_labels": _labels.tolist(),
                "system_predictions": [p for p in _predictions.tolist()],
            },
        }, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
