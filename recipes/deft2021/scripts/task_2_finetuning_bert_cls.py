#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Status     Validated 15/06/2023 Yanis LABRAK
# Apache 2.0

import os
import shutil

import uuid
import json
import logging

import numpy as np
from datasets import load_dataset, load_from_disk

from utils import parse_args, TrainingArgumentsWithMPSSupport

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, roc_auc_score, accuracy_score, classification_report
from transformers import AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, Trainer, TrainingArguments, TextClassificationPipeline

THRESHOLD_VALUE = 0.70

def toLogits(predictions, threshold=THRESHOLD_VALUE):

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    return y_pred

def multi_label_metrics(predictions, labels, threshold=THRESHOLD_VALUE):

    y_pred = toLogits(predictions, threshold)
    y_true = labels

    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {'f1_macro': f1_macro_average, 'f1_micro': f1_micro_average,  'f1_weighted': f1_weighted_average, 'accuracy': accuracy, 'roc': roc_auc}

    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids
    )
    return result

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
            "DrBenchmark/DEFT2021",
            name=args.subset,
            trust_remote_code=True,
        )

    labels_list = dataset["train"].features["specialities"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list), problem_type="multi_label_classification")

    def preprocess_function(e):
        res = tokenizer(e['text'], truncation=True, max_length=args.max_position_embeddings, padding="max_length")
        res["labels"] = e["specialities_one_hot"]
        return res

    dataset_train = dataset["train"].map(preprocess_function, batched=False).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(range(int(len(dataset_train) * args.fewshot)))
    if args.max_train_samples:
        dataset_train = dataset_train.select(range(args.max_train_samples))
    dataset_train = dataset_train.remove_columns(["text","id","specialities"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    if args.max_val_samples:
        dataset_val = dataset_val.select(range(args.max_val_samples))
    dataset_val = dataset_val.remove_columns(["text","id","specialities"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    if args.max_test_samples:
        dataset_test = dataset_test.select(range(args.max_test_samples))
    dataset_test = dataset_test.remove_columns(["text","id","specialities"])
    # true_labels = list(dataset_test["labels"])
    dataset_test.set_format("torch")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-DEFT2021-{args.subset}-{str(uuid.uuid4().hex)}"

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
        metric_for_best_model="f1_weighted",
        push_to_hub=False,
        save_only_model=True,
        save_total_limit=1,
        report_to='none',
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

    predictions, labels, _ = trainer.predict(dataset_test)

    predictions = toLogits(predictions, THRESHOLD_VALUE)

    metrics = multi_label_metrics(predictions, labels, THRESHOLD_VALUE)
    print(metrics)

    cr = classification_report(labels, predictions, labels=range(len(labels_list)), target_names=labels_list, digits=4)
    print(cr)

    with open(f"../runs/{output_name}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": f"{args.output_dir}/{output_name}_best_model",
            "metrics": classification_report(labels, predictions, output_dict=True),
            "hyperparameters": vars(args),
            "predictions": {
                "identifiers": dataset["test"]["id"],
                "real_labels": labels.tolist(),
                "system_predictions": predictions.tolist(),
            },
        }, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
