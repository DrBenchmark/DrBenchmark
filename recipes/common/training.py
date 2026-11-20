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
import platform
import subprocess
import dataclasses
import importlib.metadata

import numpy as np
import torch
from transformers import set_seed
from packaging.version import parse as parse_version
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import __version__ as transformers_version

from . import utils

transformers_version = parse_version(transformers_version)

# The libraries whose version is written in the results of a run: the ones that train and evaluate
LIBRARIES = ("torch", "transformers", "tokenizers", "datasets", "accelerate", "evaluate", "numpy", "scipy", "scikit-learn", "seqeval")

# The file, next to a model that a run kept, which says what the model is: `predict.py` reads it
DESCRIPTION = "drbenchmark.json"

# What separates the labels of an example, when it has several, in a file
SEPARATOR = "|"


class Task:
    """A task of the benchmark, as the script of its recipe defines it.

    The script creates a task with what is specific to it, and trains a model for it:
    `TASK.train()`. `predict.py` loads the same script to predict with a model that a run
    kept: what the model reads is built by the same code when it is trained and when it predicts.

    A prediction is what a person reads: the labels of the words of a sentence, the name of a
    class, the names of several classes, a score, or the letters of the right answers.
    """

    #: Name of the kind of task
    kind = None
    #: Class of `transformers` that loads a model of the task
    model_class = None
    #: The columns of an example that the model reads, in their order: {column: "text" or "words"}
    inputs = {}
    #: The column of an annotated example that has what the model should predict
    reference_column = None
    #: The column that has the identifier of an example
    id_column = "id"
    #: Whether a prediction is one label for each word of the column `tokens`
    word_labels = False
    #: The names of the classes, for a task whose classes are not the ones of a column of its dataset
    labels = None

    def train(self):
        """Fine-tune a model, evaluate it on the test set and write the results of the run."""
        raise NotImplementedError

    def encode(self, example, tokenizer, max_length, model_name):
        """What is given to the model for an example: its arguments in `inputs`, and what `decode` needs.

        `model_name` is the name of the model that is fine-tuned, as `model_of` gives it.
        """
        raise NotImplementedError

    def decode(self, logits, encoded, labels):
        """The prediction for an example, from the output of the model for what `encode` returned."""
        raise NotImplementedError

    def reference(self, example, labels, features=None):
        """What a model should predict for an annotated example, in the form of a prediction.

        The classes of the example are numbers in a dataset, whose `features` name them, or
        names when the example is read from a file: `labels` are the names of the outputs of the model.
        """
        raise NotImplementedError

    def metrics(self, references, predictions, labels, identifiers=None):
        """The scores of predictions against references: the ones that a run computes on the test set."""
        raise NotImplementedError


def model_of(args):
    """Name of the model that a run fine-tunes: the one of the HuggingFace Hub when it is known, or else its path."""
    return args.hub_model_name or args.model_name


def name_outputs(model, labels):
    """Give the outputs of a model the names of the classes of its task: they are saved with the model."""
    model.config.id2label = dict(enumerate(labels))
    model.config.label2id = {label: index for index, label in enumerate(labels)}


def class_names(features, column):
    """Names of the classes of a column of a dataset: None when the column is not one of classes, or is not known."""
    feature = features.get(column) if features else None
    if isinstance(feature, list):
        # A list of classes, for datasets<4
        feature = feature[0] if feature else None
    for candidate in (feature, getattr(feature, "feature", None)):
        names = getattr(candidate, "names", None)
        if names is not None:
            return list(names)
    return None


def name_of(value, names):
    """Name of a class that is given by its number, as in a dataset, or by its name, as in a file."""
    if isinstance(value, str):
        if value in names:
            return value
    elif 0 <= int(value) < len(names):
        return names[int(value)]
    raise ValueError(f"Unknown class {value!r}: the classes are {', '.join(repr(name) for name in names)}")


class Predictor:
    """A model that a run kept, loaded to predict.

    Args:
        task: the task of the model.
        directory: where the model and its tokenizer were saved.
        max_length: number of sub-word tokens the texts are cut at: the one of the run that trained the model.
        model_name: name of the model that was fine-tuned, as `model_of` gives it.
    """

    def __init__(self, task, directory, max_length=None, model_name=None):
        self.task = task
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        # Without the name of the model that was fine-tuned, the one of its tokenizer says which model it is
        self.model_name = model_name or type(self.tokenizer).__name__
        self.model = task.model_class.from_pretrained(directory)
        self.model.eval()
        # The names that the run gave to the outputs of the model
        self.labels = [self.model.config.id2label[index] for index in range(self.model.config.num_labels)]
        unnamed = self.labels == [f"LABEL_{index}" for index in range(len(self.labels))]
        if unnamed and task.labels is not None and len(task.labels) == len(self.labels):
            # A model that the first scripts saved without these names, for a task that knows them without its dataset
            self.labels = list(task.labels)

    def __call__(self, examples):
        """Predictions for examples: dictionaries that have the columns of `task.inputs`."""
        predictions = []
        with torch.no_grad():
            for example in examples:
                encoded = self.task.encode(example, self.tokenizer, self.max_length, self.model_name)
                inputs = {name: torch.tensor([values]) for name, values in encoded["inputs"].items()}
                logits = self.model(**inputs).logits[0]
                predictions.append(self.task.decode(logits.numpy(), encoded, self.labels))
        return predictions


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
    # The task that is trained, and the names of the outputs of its model
    task: object = None
    labels: object = None


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
            f"Save it with `python {os.path.join(utils.REPO_DIR, 'download_models_locally.py')} --models {args.hub_model_name}`, {hub}."
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


def start(dataset_name, needs_data_dir=False, task=None):
    """Parse the command line, fix the seed and load the dataset, for the run of a `task`.

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

    return Run(args=args, dataset=dataset, name=name, path=path, model_path=model_path, task=task)


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


def code_version():
    """Commit of the repository, and whether its files were changed since: None when they are not known.

    They are not known when the repository was not cloned with git, or when git is not installed.
    """
    def git(*arguments):
        completed = subprocess.run(["git", "-C", utils.REPO_DIR, *arguments], capture_output=True, text=True, timeout=60)
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit, modified = None, None
    try:
        top_level = git("rev-parse", "--show-toplevel")
        # A copy of the scripts in the repository of something else has no commit of its own
        if top_level is not None and os.path.realpath(top_level) == utils.REPO_DIR:
            commit = git("rev-parse", "HEAD")
            changes = git("status", "--porcelain", "--untracked-files=no")
            modified = None if changes is None else bool(changes)
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": commit, "modified": modified}


def environment():
    """What the run is computed with: the versions of Python, of the libraries and of the code of the repository.

    It is written in the results of the run. Two runs that have the same arguments may not have
    been computed in the same way: by other versions of the libraries, or of the scripts.
    """
    versions = {"python": platform.python_version()}
    for library in LIBRARIES:
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = None
    return {**versions, **code_version()}


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def main_scores(metrics):
    """The scores of a run that are numbers, and the averaged F1 of a classification: {name: score}."""
    scores = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[name] = value
        elif isinstance(value, dict) and name in ("macro avg", "weighted avg") and "f1-score" in value:
            scores[f"{name} f1-score"] = value["f1-score"]
    return scores


def validation_score(trainer):
    """The score that chose the checkpoint which is kept: it is computed on the validation set."""
    return {"metric": trainer.args.metric_for_best_model, "score": trainer.state.best_metric}


def describe_model(run, trainer, metrics, directory):
    """Write, next to the model that the run keeps, what it is: for `predict.py` and for the reader.

    - `drbenchmark.json`: the task, the script that builds what the model reads, the names of
      its outputs and the scores of the run. With it the directory is enough to predict.
    - `README.md`: the same as a model card, which the HuggingFace Hub shows when the
      directory is published there.
    """
    args, task = run.args, run.task
    script = os.path.realpath(sys.argv[0])
    if utils.is_in_repository(script):
        script = os.path.relpath(script, utils.REPO_DIR)
    model = model_of(args)
    # A model that is not one of the HuggingFace Hub is a directory: its name is enough for a reader
    shown_model = args.hub_model_name or os.path.basename(os.path.normpath(args.model_name))
    validation = validation_score(trainer)
    description = {
        "task": args.task,
        "kind": task.kind,
        "dataset": args.dataset,
        "subset": args.subset,
        "model": model,
        "seed": args.seed,
        "script": script,
        "inputs": task.inputs,
        "labels": run.labels,
        "max_length": args.max_position_embeddings,
        "hyperparameters": {
            name: getattr(args, name)
            for name in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay", "fewshot")
        },
        "validation": validation,
        "metrics": metrics,
        "environment": environment(),
    }
    with open(os.path.join(directory, DESCRIPTION), "w", encoding="utf-8") as f:
        json.dump(description, f, ensure_ascii=False, indent=4, default=np_encoder)

    header = ["language: fr", "library_name: transformers", "tags:", "- drbenchmark", "- biomedical"]
    tagging = task.model_class.__name__.endswith("TokenClassification")
    header.append(f"pipeline_tag: {'token-classification' if tagging else 'text-classification'}")
    if args.hub_model_name:
        header.append(f"base_model: {args.hub_model_name}")
    if str(args.dataset).startswith("DrBenchmark/") and not getattr(task, "needs_data_dir", False):
        # A dataset of the HuggingFace Hub
        header += ["datasets:", f"- {args.dataset}"]
    scores = "\n".join(f"| {name} | {score:.4f} |" for name, score in main_scores(metrics).items())
    hyperparameters = "\n".join(f"- {name}: {value}" for name, value in description["hyperparameters"].items())
    inputs = ", ".join(f"`{name}` ({'the words of a sentence' if kind == 'words' else 'a text'})" for name, kind in task.inputs.items())
    newline = "\n"
    card = f"""---
{newline.join(header)}
---

# {shown_model} fine-tuned on {args.task}

This model is `{shown_model}` fine-tuned on the task `{args.task}` of [DrBenchmark](https://github.com/DrBenchmark/DrBenchmark),
a benchmark of French biomedical language understanding: {task.kind}, on the dataset `{args.dataset}`{f" (subset `{args.subset}`)" if args.subset else ""}.

It is the checkpoint that one run kept, with the seed {args.seed}: the one that has the best `{validation['metric']}`
on the validation set ({validation['score']:.4f}).

## Scores on the test set

| Metric | Score |
|:-------|------:|
{scores}

## Training

{hyperparameters}
- at most {args.max_position_embeddings} sub-word tokens per example

## What the model reads

The model reads {inputs}. The script that says how they are given to it is `{script}`:
another tool, such as a `pipeline` of `transformers` or the widget of the HuggingFace Hub, gives
the model the text as it is typed. `{DESCRIPTION}` has what is written here for programs.

## Predicting

With the repository of DrBenchmark, which builds what the model reads as when it was trained:

```bash
python predict.py --model-dir <this directory> "..."
```
"""
    with open(os.path.join(directory, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def save_results(run, trainer, metrics, identifiers, real_labels, system_predictions):
    """Write the results of the run: they are read by `tools/CollectAndAverageScores.py`.

    The best model is kept with what describes it, or removed when the run was asked not to keep it.
    """
    best_model = run.model_path + "_best_model"
    if not run.args.keep_model:
        shutil.rmtree(best_model, ignore_errors=True)
    elif run.task is not None:
        describe_model(run, trainer, metrics, best_model)
    with open(run.path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": best_model if run.args.keep_model else None,
            "metrics": metrics,
            "hyperparameters": vars(run.args),
            "validation": validation_score(trainer),
            "environment": environment(),
            "predictions": {
                "identifiers": identifiers,
                "real_labels": real_labels,
                "system_predictions": system_predictions,
            },
            'trainer_state': dataclasses.asdict(trainer.state),
        }, f, ensure_ascii=False, indent=4, default=np_encoder)
    utils.report_status(utils.COMPUTED, run.path)
