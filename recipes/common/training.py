#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Steps shared by the fine-tuning scripts of every kind of task."""

import os
import sys
import json
import shutil
import logging
import dataclasses

import numpy as np
from transformers import set_seed
from packaging.version import parse as parse_version
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments
from transformers import __version__ as transformers_version

from . import utils

transformers_version = parse_version(transformers_version)


@dataclasses.dataclass
class Run:
    """What a script needs to know about the run it is doing."""
    args: object
    # Dataset with its three splits: train, validation and test
    dataset: object
    # Name of the run, built from its arguments
    name: str
    # File where the results of the run are written
    path: str
    # Directory of the checkpoints; the best model is saved in `<model_path>_best_model`
    model_path: str


def local_dataset(args):
    """Directory of the copy of the dataset that is read in offline mode."""
    return f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/"


def check_local_copies(args, needs_data_dir=False):
    """In offline mode, stop the script when the copy of its dataset or of its model is missing, and say what to do.

    Otherwise `load_from_disk` and `from_pretrained` fail on a directory that they do not
    find, without a word about the scripts that save these copies.
    """
    hub = "on a machine that has access to the HuggingFace Hub"
    missing = []
    # What can be read from the Hub instead
    on_the_hub = []

    if not os.path.isdir(local_dataset(args)):
        subset = f" (subset {args.subset})" if args.subset else ""
        message = f"The dataset {args.dataset}{subset} is not in {os.path.abspath(local_dataset(args))}, where it is read in offline mode.\n"
        if needs_data_dir:
            # DEFT-2019: its access is restricted
            message += "It is not on the HuggingFace Hub and `download_datasets_locally.py` does not save it: see the README of its recipe."
        else:
            message += f"Save the datasets with `python {os.path.join(utils.REPO_DIR, 'download_datasets_locally.py')}`, {hub}."
            on_the_hub.append("the dataset")
        missing.append(message)

    # `hub_model_name` is the model when it is not a path that exists: its copy is read in `models/`
    if args.hub_model_name is not None and not os.path.isdir(args.model_name):
        missing.append(
            f"The model {args.hub_model_name} is not a directory, and its copy is not in {os.path.abspath(args.model_name)}, "
            "where it is read in offline mode.\n"
            f"Save the models of models.txt with `python {os.path.join(utils.REPO_DIR, 'download_models_locally.py')}`, {hub}."
        )
        on_the_hub.append("the model")

    if not missing:
        return
    if on_the_hub:
        missing.append(
            f"To read {' and '.join(on_the_hub)} from the HuggingFace Hub instead, give `--offline false` "
            f"or set `offline: False` in {os.path.join(utils.REPO_DIR, 'config.yaml')}."
        )
    sys.exit("\n\n".join(missing))


def start(dataset_name, needs_data_dir=False):
    """Parse the command line, fix the seed and load the dataset.

    In offline mode, the dataset is read from `<data_dir>/local_hf_<subset>/`: the script stops,
    and says which script saves them, when this copy or the one of the model is missing.
    Otherwise `dataset_name` is given to `datasets.load_dataset`, with `--data_dir`
    when `needs_data_dir` is set (loading script that reads local files).
    """
    args = utils.parse_args()
    # The names are computed now: they depend on the arguments, which a script may change later
    name = utils.create_run_name(args)
    path = utils.get_run_path(args)
    model_path = utils.get_model_path(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # The script may have been started from another directory: the relative paths of its
    # arguments, of its results and of its messages are relative to this one
    logging.info(f"Working directory: {os.getcwd()}")

    set_seed(args.seed)
    if args.offline:
        check_local_copies(args, needs_data_dir)
        dataset = load_from_disk(local_dataset(args))
    else:
        kwargs = {"data_dir": args.data_dir} if needs_data_dir else {}
        dataset = load_dataset(dataset_name, name=args.subset, **kwargs)

    return Run(args=args, dataset=dataset, name=name, path=path, model_path=model_path)


def prepare_splits(run, function, **map_kwargs):
    """Apply `function` to the three splits and select the examples that are used.

    The training set is shuffled, always in the same way, then cut according to
    `--fewshot` and `--max_train_samples`.
    """
    args, dataset = run.args, run.dataset

    train = dataset["train"].map(function, **map_kwargs).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        train = train.select(range(int(len(train) * args.fewshot)))
    if args.max_train_samples:
        train = train.select(range(args.max_train_samples))

    validation = dataset["validation"].map(function, **map_kwargs)
    if args.max_val_samples:
        validation = validation.select(range(args.max_val_samples))

    test = dataset["test"].map(function, **map_kwargs)
    if args.max_test_samples:
        test = test.select(range(args.max_test_samples))

    return train, validation, test


def training_arguments(run, metric_for_best_model, **kwargs):
    """Arguments of the Trainer: one evaluation per epoch, and the best checkpoint is kept."""
    args = run.args
    os.makedirs(args.output_dir, exist_ok=True)

    return TrainingArguments(
        run.model_path,
        **{'eval_strategy' if transformers_version >= parse_version('4.41') else 'evaluation_strategy': 'epoch'},
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # The weights are updated every `gradient_accumulation_steps` batches:
        # the effective batch size is `batch_size * gradient_accumulation_steps`
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        push_to_hub=False,
        save_only_model=True,
        save_total_limit=1,
        report_to='none',
        **kwargs,
    )


def create_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, compute_metrics, **kwargs):
    return Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **{'processing_class' if transformers_version >= parse_version('4.46') else 'tokenizer': tokenizer},
        compute_metrics=compute_metrics,
        **kwargs,
    )


def train(run, trainer):
    """Train, then replace the checkpoints by the best model."""
    logging.info("***** Starting Training *****")
    trainer.train()
    trainer.evaluate()

    logging.info("***** Save the best model *****")
    trainer.save_model(run.model_path + "_best_model")
    shutil.rmtree(run.model_path)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def save_results(run, trainer, metrics, identifiers, real_labels, system_predictions):
    """Write the results of the run: they are read by `tools/CollectAndAverageScores.py`."""
    with open(run.path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": run.model_path + "_best_model",
            "metrics": metrics,
            "hyperparameters": vars(run.args),
            "predictions": {
                "identifiers": identifiers,
                "real_labels": real_labels,
                "system_predictions": system_predictions,
            },
            'trainer_state': dataclasses.asdict(trainer.state),
        }, f, ensure_ascii=False, indent=4, default=np_encoder)
