#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

import os
import json
import shutil
import logging
import dataclasses

from transformers import set_seed
from packaging.version import parse as parse_version
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import __version__ as transformers_version
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import utils

transformers_version = parse_version(transformers_version)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=.0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():

    args = utils.parse_args()
    run_name = utils.create_run_name(args)
    run_path = utils.get_run_path(args)
    model_path = utils.get_model_path(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(args.seed)
    if args.offline:
        dataset = load_from_disk(f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/")
    else:
        dataset = load_dataset(
            "DrBenchmark/FrenchMedMCQA",
        )

    labels_list = ["c", "a", "e", "d", "b", "be", "ae", "bc", "bd", "ab", "de", "cd", "ac", "ad", "ce", "bce", "abc", "cde", "bcd", "ace", "ade", "abe", "acd", "bde", "abd", "abde", "abcd", "bcde", "abce", "acde", "abcde"]
    original_labels_list = dataset["train"].features["correct_answers"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(labels_list))

    def preprocess_function(e):

        concatenated_choices = f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in ["a", "b", "c", "d", "e"]])
        text = f"{tokenizer.cls_token} {e['question']} {tokenizer.sep_token} {concatenated_choices} {tokenizer.eos_token}"

        res = tokenizer(text, truncation=True, max_length=args.max_position_embeddings, padding="max_length")
        res["text"] = text

        labels_letters = [original_labels_list[l] for l in e["correct_answers"]]  # Get from the answers indexes to the original letters
        labels_letters = "".join(sorted(labels_letters))  # Transform the multi-label answers into a multi-class problem
        labels_letters = labels_list.index(labels_letters)  # Get the corresponding index of the class
        res["label"] = labels_letters

        return res

    metric_name = "accuracy"

    dataset_train = dataset["train"].map(preprocess_function, batched=False).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(range(int(len(dataset_train) * args.fewshot)))
    if args.max_train_samples:
        dataset_train = dataset_train.select(range(args.max_train_samples))
    dataset_train = dataset_train.remove_columns(["text"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    if args.max_val_samples:
        dataset_val = dataset_val.select(range(args.max_val_samples))
    dataset_val = dataset_val.remove_columns(["text"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    if args.max_test_samples:
        dataset_test = dataset_test.select(range(args.max_test_samples))

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        model_path,
        **{'eval_strategy' if transformers_version >= parse_version('4.41') else 'evaluation_strategy': 'epoch'},
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
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
        **{'processing_class' if transformers_version >= parse_version('4.46') else 'tokenizer': tokenizer},
        compute_metrics=compute_metrics,
    )

    logging.info("***** Starting Training *****")
    trainer.train()
    trainer.evaluate()

    logging.info("***** Save the best model *****")
    trainer.save_model(model_path + "_best_model")
    shutil.rmtree(model_path)

    logging.info("***** Starting Evaluation *****")
    pipe = pipeline(task="text-classification", model=model_path + "_best_model", top_k=None, device=training_args.device)

    def compute_accuracy_exact_match(preds, refs):
        exact_score = []
        for p, r in zip(preds, refs):
            exact_score.append(sorted(p) == sorted(r))
        return sum(exact_score) / len(exact_score)

    def compute_accuracy_hamming(preds, refs):
        corrects = [True for p in preds if p in refs]
        corrects = sum(corrects)
        total_refs = len(list(set(preds + refs)))
        return corrects / total_refs

    y_pred = []
    y_true = []

    hamming_scores = []

    for e in dataset_test:

        res = pipe(e["text"], truncation=True, max_length=model.config.max_position_embeddings)

        pred = int(res[0][0]["label"].split("_")[-1])
        pred = labels_list[pred]
        y_pred.append(pred)
        splitted_pred = sorted(list(pred))

        # Reference
        true_label = labels_list[e["label"]]
        y_true.append(true_label)
        splitted_true_label = sorted(list(true_label))

        # Compute hamming score
        score = compute_accuracy_hamming(splitted_pred, splitted_true_label)
        hamming_scores.append(score)

    logging.info(">> Hamming Score")
    hamming_score = sum(hamming_scores) / len(hamming_scores)
    logging.info(hamming_score)

    logging.info(">> Exact Match Ratio (EMR)")
    exact_match = compute_accuracy_exact_match(y_true, y_pred)
    logging.info(exact_match)

    with open(run_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_path + "_best_model",
            "metrics": {
                "hamming_score": float(hamming_score),
                "exact_match": float(exact_match),
            },
            "hyperparameters": vars(args),
            "predictions": {
                "identifiers": list(dataset["test"]["id"]),
                "real_labels": y_true,
                "system_predictions": y_pred,
            },
            'trainer_state': dataclasses.asdict(trainer.state),
        }, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
