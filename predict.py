#!/usr/bin/env python3
"""Predict with a model that a run of the benchmark kept: on a sentence, on a file, or on annotated data.

    python predict.py --list
    python predict.py --task cas-pos --model Dr-BERT/DrBERT-7GB "Le patient présente une toux sèche."
    python predict.py --task cas-pos --model Dr-BERT/DrBERT-7GB
    python predict.py --task cas-pos --model Dr-BERT/DrBERT-7GB --input sentences.txt --output labels.conll
    python predict.py --task cas-pos --model Dr-BERT/DrBERT-7GB --evaluate corpus.conll
    python predict.py --model-dir recipes/cas/save_models/<run>_best_model "Le patient présente une toux sèche."

A run keeps its best model in `recipes/<recipe>/save_models/` and writes its results in
`recipes/<recipe>/runs/`. The model is the one of a task (`--task`, a name of `python run.py
--help`) for a model that was fine-tuned (`--model`, as it was given to `run.py`). When
several runs kept their model, the one with the best score on the validation set is taken,
or the one of `--seed`. `--list` shows the runs, their scores and the one that is taken.
`--model-dir` gives the directory of a kept model instead, wherever it is.

What the model reads is built by the script of the task, as when the model was trained.

Without a text, `--input` or `--evaluate`, the texts are asked for, until an empty line.

`--input` and `--evaluate` read:
- a text file: one example per line, with a tabulation between the texts of a task that reads several;
- a CoNLL file (`.conll`, or `--format conll`), for the tasks that label words: one word per
  line, an empty line after each sentence. `--evaluate` reads the label of a word at the end of its line;
- a CSV or TSV file (`.csv`, `.tsv`, or `--format`), for the other tasks: its first line names its
  columns, which are the ones of the dataset of the task. `--evaluate` reads the column of the answer;
- a directory: a dataset saved by `datasets`, with the columns of the dataset of the task. The
  copies of `download_datasets_locally.py` are such directories.

The predictions of `--input` are written on the standard output, or in the file of `--output`:
in the CoNLL format for the tasks that label words, in CSV for the other ones. `--evaluate`
prints the scores of the task, the ones a run computes on its test set; with `--output`, it
also writes its predictions, next to what was expected.
"""

import argparse
import csv
import dataclasses
import importlib.util
import json
import os
import re
import sys
from glob import glob

# This file is at the root of the repository
REPO_DIR = os.path.dirname(os.path.realpath(__file__))
RECIPES_DIR = os.path.join(REPO_DIR, "recipes")

# The file that a run writes next to the model it keeps, to say what the model is
DESCRIPTION = "drbenchmark.json"

FORMATS = ("text", "conll", "csv", "tsv")
# The format of a file that `--format` does not give, from the end of its name
EXTENSIONS = {".conll": "conll", ".conllu": "conll", ".iob": "conll", ".iob2": "conll", ".bio": "conll", ".csv": "csv", ".tsv": "tsv"}

# The words of a sentence, as most datasets of the benchmark split it: the punctuation is
# apart, and the article of "l'examen" is a word with its apostrophe
WORD = re.compile(
    r"aujourd['’]hui"          # one word
    r"|\d+(?:[.,]\d+)+"        # 7,5 or 10.25
    r"|\w+['’](?=\w)"          # l' d' qu'
    r"|\w+(?:-\w+)*"           # a word, with its hyphens: glairo-sanglantes
    r"|\.{2,}"                 # ...
    r"|[^\w\s]",               # a punctuation sign
    re.IGNORECASE,
)


class PredictError(Exception):
    """What is asked for cannot be done: the message says why."""


def split_words(text):
    """The words of a sentence that is typed as a text."""
    return WORD.findall(text)


def shared_code():
    """The code that the recipes share, `recipes/common/training.py`. It loads `torch`: it is only loaded to predict."""
    if RECIPES_DIR not in sys.path:
        # The scripts of the recipes import it as `common`: they add this directory too
        sys.path.insert(0, RECIPES_DIR)
    from common import training
    return training


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tasks_of_run_py():
    """The tasks of `run.py`, and the command of their launcher: {task: command}."""
    return load_module("drbenchmark_run", os.path.join(REPO_DIR, "run.py")).task2script


def script_of(task):
    """Path of the script that trains the models of a task of `run.py`: the one its launcher starts."""
    tasks = tasks_of_run_py()
    if task not in tasks:
        raise PredictError(f"{task} is not a task of run.py. Its tasks are: {', '.join(sorted(tasks))}")
    scripts_dir = os.path.join(RECIPES_DIR, task.split("-")[0], "scripts")
    launcher = os.path.join(scripts_dir, tasks[task].split()[0])
    with open(launcher, encoding="utf-8") as f:
        started = re.search(r'^python "\$scripts_dir/(\S+\.py)"', f.read(), re.M)
    if started is None:
        raise PredictError(f"{launcher} does not start a script of its directory")
    return os.path.join(scripts_dir, started.group(1))


def script_for(description):
    """Path of the script that defines the task of a model, from what describes the model.

    The description that a run writes next to its model names the script, from the root of
    the repository. It is only loaded when it is a script of the recipes of this repository:
    a model, and what describes it, may come from somewhere else. Otherwise the script is
    the one of the task in `run.py`.
    """
    script = description.get("script")
    if script is not None:
        recipes = os.path.realpath(RECIPES_DIR)
        script = os.path.realpath(os.path.join(REPO_DIR, script))
        if script.endswith(".py") and os.path.isfile(script) and os.path.commonpath([script, recipes]) == recipes:
            return script
    return script_of(description["task"])


def load_task(script):
    """The task that a script of a recipe defines: what is specific to it, to train and to predict."""
    shared_code()
    name = "drbenchmark_task_" + re.sub(r"\W", "_", os.path.relpath(script, RECIPES_DIR))
    module = load_module(name, script)
    if not hasattr(module, "TASK"):
        raise PredictError(f"{script} does not define its task: a line `TASK = ...`, as the scripts of recipes/ do")
    return module.TASK


# ---------------------------------------------------------------------------------------------
# The runs, and the models that they kept
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Run:
    """A run, from its results."""
    path: str
    task: str
    model: str
    seed: int
    # Proportion of the training set that was used
    fewshot: float
    hyperparameters: dict
    metrics: dict
    # Score on the validation set of the checkpoint that the run kept, and the name of this score when it is known
    validation: float
    validation_metric: str
    lower_is_better: bool
    # Directory of the model that the run kept: None when it is not there
    model_dir: str
    # Why the model is not there
    missing: str = None


def read_run(path):
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    hyperparameters = results["hyperparameters"]
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "scripts")

    state = results.get("trainer_state") or {}
    validation = results.get("validation") or {}
    metric = validation.get("metric")
    # The error of a similarity score is the only score of the benchmark that is better when it is lower
    lower_is_better = metric == "rmse" if metric else any("eval_rmse" in entry for entry in state.get("log_history") or [])

    model_dir, missing = None, None
    if results.get("model_name") is None:
        missing = "not kept (--keep_model false)"
    else:
        # The run wrote the path from the directory of its script
        model_dir = os.path.normpath(os.path.join(scripts_dir, results["model_name"]))
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            model_dir, missing = None, "not here"

    return Run(
        path=path,
        task=hyperparameters["task"],
        model=str(hyperparameters.get("hub_model_name") or hyperparameters["model_name"]),
        seed=hyperparameters["seed"],
        fewshot=hyperparameters.get("fewshot") if hyperparameters.get("fewshot") is not None else 1.0,
        hyperparameters=hyperparameters,
        metrics=results["metrics"],
        validation=validation.get("score", state.get("best_metric")),
        validation_metric=metric,
        lower_is_better=lower_is_better,
        model_dir=model_dir,
        missing=missing,
    )


def is_model(run, model):
    """Whether a run is the one of a model, given by its name on the HuggingFace Hub or by its path.

    A run in offline mode has the directory of the copy of its model, `models/<organisation>_<name>`
    in lower case, and its name on the Hub when it was computed by the recent scripts.
    """
    path = str(run.hyperparameters.get("model_name") or "")
    local_copy = os.path.basename(path.rstrip("/"))
    return model in (run.hyperparameters.get("hub_model_name"), path) or local_copy == model.lower().replace("/", "_")


def read_runs(task=None, model=None):
    """The runs whose results are in `recipes/<recipe>/runs/`, sorted by task, model and seed."""
    recipe = task.split("-")[0] if task else "*"
    runs = [read_run(path) for path in sorted(glob(os.path.join(RECIPES_DIR, recipe, "runs", "*.json")))]
    runs = [run for run in runs if (task is None or run.task == task) and (model is None or is_model(run, model))]
    return sorted(runs, key=lambda run: (run.task, run.model, run.fewshot != 1.0, run.fewshot, run.seed))


def best_run(runs):
    """The run whose kept model is taken among the ones of a task for a model: the best one on the validation set.

    The runs on a part of the training set are not taken. None when no run kept its model.
    """
    candidates = [run for run in runs if run.model_dir is not None and run.fewshot == 1.0]
    if not candidates:
        return None

    def rank(run):
        if run.validation is None:
            return (1, 0, run.seed)
        return (0, run.validation if run.lower_is_better else -run.validation, run.seed)

    return min(candidates, key=rank)


def main_scores(metrics):
    """The scores of a run that the tables of the benchmark show: {name: score}."""
    if "overall_f1" in metrics:
        return {"f1": metrics["overall_f1"], "accuracy": metrics["overall_accuracy"]}
    if "macro avg" in metrics:
        return {"macro_f1": metrics["macro avg"]["f1-score"], "weighted_f1": metrics["weighted avg"]["f1-score"]}
    if "hamming_score" in metrics:
        return {"hamming": metrics["hamming_score"], "exact_match": metrics["exact_match"]}
    if "EDRM" in metrics:
        return {"edrm": metrics["EDRM"], "spearman": metrics["spearman_correlation_coef"]}
    return {name: value for name, value in metrics.items() if isinstance(value, float)}


def number(value):
    return "-" if value is None else f"{value:.4f}"


def list_runs(task=None, model=None):
    """A table of the runs: their scores, whether their model is there, and the one that is taken for each model."""
    runs = read_runs(task, model)
    if not runs:
        return f"No run in {os.path.join(RECIPES_DIR, task.split('-')[0] if task else '*', 'runs')}" \
            + (f" for the task {task}" if task else "") + (f" for the model {model}" if model else "") + "."

    taken = set()
    for name in sorted({(run.task, run.model) for run in runs}):
        best = best_run([run for run in runs if (run.task, run.model) == name])
        if best is not None:
            taken.add(best.path)

    few_shot = any(run.fewshot != 1.0 for run in runs)
    header = ["task", "model", "seed"] + (["train"] if few_shot else []) + ["validation", "test", "kept model"]
    rows = []
    for run in runs:
        validation = number(run.validation) + (f" ({run.validation_metric})" if run.validation_metric and run.validation is not None else "")
        test = ", ".join(f"{name} {number(score)}" for name, score in main_scores(run.metrics).items())
        kept = ("yes, taken" if run.path in taken else "yes") if run.model_dir is not None else run.missing
        rows.append([run.task, run.model, str(run.seed)] + ([f"{run.fewshot:g}"] if few_shot else []) + [validation, test, kept])

    widths = [max(len(row[column]) for row in [header] + rows) for column in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [header] + rows]
    lines.append("")
    lines.append("validation: score, on the validation set, of the checkpoint that the run kept. For a task and a model, "
                 "the model that is taken is the one of the run with the best one, among the runs on the whole training set.")
    return "\n".join(lines)


def chosen_run(task, model, seed=None):
    """The run of a task for a model whose kept model is used, and the runs it was chosen among."""
    runs = [run for run in read_runs(task, model) if run.fewshot == 1.0]
    if not runs:
        others = sorted({run.model for run in read_runs(task)})
        raise PredictError(
            f"No run of {task} for {model} in {os.path.join(RECIPES_DIR, task.split('-')[0], 'runs')}.\n"
            + ("The models that have runs of this task: " + ", ".join(others) if others else "No model has a run of this task.")
            + f"\nA run is computed with `python run.py --tasks {task} --models {model}`."
        )
    names = sorted({run.model for run in runs})
    if len(names) > 1:
        raise PredictError(f"{model} is the name of several models that have runs of {task}: {', '.join(names)}")

    if seed is not None:
        with_seed = [run for run in runs if run.seed == seed]
        if not with_seed:
            raise PredictError(f"No run of {task} for {model} has the seed {seed}: the seeds of its runs are "
                               + ", ".join(str(run.seed) for run in runs) + ".")
        run = with_seed[0]
        if run.model_dir is None:
            raise PredictError(f"The model of the run of {task} for {model} with the seed {seed} is {run.missing}: {run.path}")
        return run, runs

    run = best_run(runs)
    if run is None:
        raise PredictError(
            f"No run of {task} for {model} kept its model:\n"
            + "\n".join(f"  seed {run.seed}: {run.missing} ({run.path})" for run in runs)
            + "\nA run keeps its model unless it is given `--keep_model false`."
        )
    return run, runs


# ---------------------------------------------------------------------------------------------
# A model that is loaded
# ---------------------------------------------------------------------------------------------

class Model:
    """A model that a run kept, with the task it was trained for.

    Attributes:
        name: the task of `run.py`, e.g. `cas-pos`.
        task: what the script of the task defines.
        labels: the names of the outputs of the model.
        description: what is known of the model: the model that was fine-tuned, the seed, the scores.
    """

    def __init__(self, name, task, predictor, description):
        self.name = name
        self.task = task
        self.predictor = predictor
        self.labels = predictor.labels
        self.description = description

    def example(self, *texts, words=False):
        """An example of the task, from the texts that it reads, in the order of `task.inputs`.

        A text which is the words of a sentence is split into words: at its spaces when
        `words` says that it is already split, as the datasets of the benchmark are otherwise.
        """
        columns = self.task.inputs
        if len(texts) != len(columns):
            raise PredictError(
                f"{self.name} reads {len(columns)} text{'s' if len(columns) > 1 else ''} ({', '.join(columns)}): {len(texts)} given")
        return {
            column: (text.split() if words else split_words(text)) if kind == "words" else text
            for (column, kind), text in zip(columns.items(), texts)
        }

    def predict(self, examples):
        """The predictions for examples: dictionaries that have the columns of `task.inputs`."""
        return self.predictor(examples)


def load(task=None, model=None, seed=None, model_dir=None):
    """Load the model that a run kept: the one of a task for a model, or the one of a directory."""
    description = {}
    if model_dir is not None:
        if not os.path.isfile(os.path.join(model_dir, "config.json")):
            raise PredictError(f"{model_dir} is not the directory of a model: it has no config.json")
    else:
        if task is None or model is None:
            raise PredictError("give the task and the model that was fine-tuned, or the directory of a kept model")
        run, runs = chosen_run(task, model, seed)
        model_dir = run.model_dir
        description = {
            "task": run.task, "model": run.model, "seed": run.seed, "max_length": run.hyperparameters.get("max_position_embeddings"),
            "validation": {"metric": run.validation_metric, "score": run.validation},
            "metrics": run.metrics, "results": run.path,
            # The number of runs the model was chosen among: one when its seed is given
            "chosen_among": 1 if seed is not None else len([other for other in runs if other.model_dir is not None]),
        }

    # What the run wrote next to its model
    written = os.path.join(model_dir, DESCRIPTION)
    if os.path.isfile(written):
        with open(written, encoding="utf-8") as f:
            description = {**description, **json.load(f)}
    elif task is not None:
        description.setdefault("task", task)
    if "task" not in description:
        raise PredictError(
            f"{model_dir} has no {DESCRIPTION}, which says what the task of the model is: "
            "it was kept by an earlier version of the scripts. Give its task with --task.")

    # Once the model is found: what predicts is slow to load
    script = script_for(description)
    training = shared_code()
    task_of_the_model = load_task(script)
    predictor = training.Predictor(task_of_the_model, model_dir, max_length=description.get("max_length"), model_name=description.get("model"))
    description["directory"] = model_dir
    return Model(description["task"], predictor.task, predictor, description)


# ---------------------------------------------------------------------------------------------
# What is read and written
# ---------------------------------------------------------------------------------------------

def spans(labels):
    """The entities of a sentence, from the labels of its words: (first word, word after the last one, type).

    A label is `B-<type>` for the first word of an entity and `I-<type>` for the next ones,
    with `-` or `_`. A label without such a mark is a type by itself. `O` is no entity.
    """
    found = []
    for index, label in enumerate(labels):
        mark, kind = (label[0], label[2:]) if len(label) > 2 and label[0] in "BI" and label[1] in "-_" else (None, label)
        if label == "O":
            continue
        continues = found and found[-1][1] == index and found[-1][2] == kind and mark != "B"
        if continues:
            found[-1] = (found[-1][0], index + 1, kind)
        else:
            found.append((index, index + 1, kind))
    return found


def words_with_labels(words, labels):
    """A sentence with its entities marked: `word/TYPE`, or `[several words]/TYPE`."""
    shown, position = [], 0
    for first, after, kind in spans(labels):
        shown.extend(words[position:first])
        entity = words[first:after]
        shown.append((entity[0] if len(entity) == 1 else "[" + " ".join(entity) + "]") + "/" + kind)
        position = after
    shown.extend(words[position:])
    return " ".join(shown)


def as_text(prediction, separator="|"):
    """A prediction, or what a column of an example has, as it is written in a file."""
    if isinstance(prediction, (list, tuple)):
        return separator.join(str(value) for value in prediction)
    if isinstance(prediction, float):
        return f"{prediction:.4f}"
    return str(prediction)


def shown(model, example, prediction):
    """A prediction as it is shown to a person."""
    if model.task.word_labels:
        return words_with_labels(example["tokens"], prediction)
    if isinstance(prediction, (list, tuple)):
        return ", ".join(prediction) if prediction else "(no label)"
    if isinstance(prediction, float):
        return f"{prediction:.2f}"
    return str(prediction)


@dataclasses.dataclass
class Examples:
    """Examples that are read: what a model is given, and the columns that are written with its predictions."""
    examples: list
    columns: list
    # The features of a dataset: they name its classes
    features: object = None


def format_of(path, given=None):
    if given is not None:
        return given
    return EXTENSIONS.get(os.path.splitext(path)[1].lower(), "text")


def read_conll(path, model, annotated):
    column = model.task.reference_column
    examples, words, labels = [], [], []

    def end_of_sentence():
        if words:
            examples.append({"tokens": list(words), **({column: list(labels)} if annotated else {})})
            words.clear()
            labels.clear()

    with open(path, encoding="utf-8") as f:
        for number_, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                end_of_sentence()
                continue
            if line.startswith("-DOCSTART-"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if annotated and len(fields) < 2:
                raise PredictError(f"{path}, line {number_}: a word and its label are expected, separated by a tabulation or a space")
            words.append(fields[0])
            labels.append(fields[-1])
    end_of_sentence()
    return Examples(examples, ["tokens"] + ([column] if annotated else []))


def read_table(path, model, annotated, words, delimiter):
    task = model.task
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        needed = list(task.inputs) + ([task.reference_column] if annotated else [])
        missing = [column for column in needed if column not in columns]
        if missing:
            raise PredictError(
                f"{path} has no column {', '.join(missing)}: its first line names its columns, which are {', '.join(columns) or 'none'}. "
                f"{model.name} reads the column{'s' if len(needed) > 1 else ''} {', '.join(needed)}.")
        examples = []
        for row in reader:
            example = dict(row)
            example.update(model.example(*(row[column] for column in task.inputs), words=words))
            examples.append(example)
    return Examples(examples, columns)


def read_text(path, model, words):
    examples = []
    with open(path, encoding="utf-8") as f:
        for number_, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            try:
                examples.append(model.example(*line.split("\t"), words=words))
            except PredictError as error:
                raise PredictError(f"{path}, line {number_}: {error}. The texts of an example are separated by tabulations.")
    return Examples(examples, list(model.task.inputs))


def read_dataset(path, model, annotated, split):
    from datasets import load_from_disk

    task = model.task
    dataset = load_from_disk(path)
    if hasattr(dataset, "keys"):
        # The splits of a dataset
        if split not in dataset:
            raise PredictError(f"{path} has no split {split}: its splits are {', '.join(dataset.keys())}. Give one with --split.")
        dataset = dataset[split]
    missing = [column for column in task.inputs if column not in dataset.column_names]
    if missing:
        raise PredictError(f"{path} has no column {', '.join(missing)}: its columns are {', '.join(dataset.column_names)}")
    columns = [column for column in [task.id_column, *task.inputs] if column in dataset.column_names]
    if annotated and task.reference_column in dataset.column_names:
        columns.append(task.reference_column)
    return Examples(list(dataset), columns, dataset.features)


def read_examples(path, model, annotated=False, words=False, split="test", file_format=None):
    """The examples of a file or of a saved dataset, for a model. `annotated`: with what the model should predict."""
    if os.path.isdir(path):
        return read_dataset(path, model, annotated, split)
    if not os.path.isfile(path):
        raise PredictError(f"{path} does not exist")

    file_format = format_of(path, file_format)
    labels_words = model.task.word_labels
    if file_format == "conll":
        if not labels_words:
            raise PredictError(f"{path}: {model.name} does not label words, its examples are not read in a CoNLL file. "
                               "They are read in a text file, or in a CSV or TSV file.")
        return read_conll(path, model, annotated)
    if labels_words and (annotated or file_format in ("csv", "tsv")):
        raise PredictError(
            f"{path}: {model.name} labels words, its {'annotated ' if annotated else ''}examples are read in a CoNLL file "
            "(`.conll`, or `--format conll`): one word per line, an empty line after each sentence"
            + (", and the label of a word at the end of its line." if annotated else "."))
    if file_format in ("csv", "tsv"):
        return read_table(path, model, annotated, words, "," if file_format == "csv" else "\t")
    if annotated:
        raise PredictError(
            f"{path}: the annotated examples of {model.name} are read in a CSV or TSV file (`.csv`, `.tsv`, or `--format`), "
            f"whose columns are {', '.join([*model.task.inputs, model.task.reference_column])}.")
    return read_text(path, model, words)


def references_of(model, read):
    """What the model should predict for annotated examples, and their identifiers."""
    task = model.task
    references = []
    for index, example in enumerate(read.examples):
        try:
            references.append(task.reference(example, model.labels, read.features))
        except KeyError as error:
            raise PredictError(f"The example {index + 1} has no column {error}: it is not annotated for {model.name}")
        except ValueError as error:
            raise PredictError(f"Example {index + 1}: {error}")
    identifiers = [example.get(task.id_column, index) for index, example in enumerate(read.examples)]
    return references, identifiers


def write_predictions(file, model, read, predictions, references=None):
    """Write predictions: in the CoNLL format for the labels of words, in CSV for the other ones."""
    task = model.task
    if task.word_labels:
        for index, (example, prediction) in enumerate(zip(read.examples, predictions)):
            for position, word in enumerate(example["tokens"]):
                fields = [word] + ([references[index][position]] if references is not None else []) + [prediction[position]]
                file.write("\t".join(fields) + "\n")
            file.write("\n")
        return

    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(read.columns + ["prediction"])
    for index, (example, prediction) in enumerate(zip(read.examples, predictions)):
        row = []
        for column in read.columns:
            if references is not None and column == task.reference_column:
                row.append(as_text(references[index]))
            else:
                row.append(as_text(example[column], " " if task.inputs.get(column) == "words" else "|"))
        writer.writerow(row + [as_text(prediction)])


def scores_as_text(metrics, labels):
    """The scores of an evaluation: the ones of the whole, then the ones of each class."""
    lines, details = [], []
    for name, value in metrics.items():
        if isinstance(value, dict):
            # The classes of a classification are written by their numbers
            name = labels[int(name)] if name.isdigit() and int(name) < len(labels) else name
            details.append(f"  {name}: " + ", ".join(
                f"{key} {score:.4f}" if isinstance(score, float) and key not in ("support", "number") else f"{key} {score:g}"
                for key, score in value.items()))
        elif isinstance(value, (int, float)):
            lines.append(f"{name}: {value:.4f}")
    return "\n".join(lines + details)


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------

def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="*", metavar="TEXT",
                        help="Texts of the examples to predict: one text per example, or as many as the task reads.")
    parser.add_argument("--list", action="store_true",
                        help="List the runs, their scores and the models that they kept. With --task or --model: only their runs.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task of the model: one of the names of `python run.py --help`.")
    parser.add_argument("--model", type=str, default=None,
                        help="Model that was fine-tuned: its name on the HuggingFace Hub, or its path.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed of the run whose model is used (default: the run with the best score on the validation set).")
    parser.add_argument("--model-dir", type=str, default=None, metavar="DIR",
                        help="Directory of a model that a run kept, instead of --task and --model.")
    parser.add_argument("--input", type=str, default=None, metavar="PATH",
                        help="File, or saved dataset, of the examples to predict.")
    parser.add_argument("--evaluate", type=str, default=None, metavar="PATH",
                        help="File, or saved dataset, of annotated examples: the predictions are scored with the metrics of the task.")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="File where the predictions are written (default: the standard output for --input, and no "
                             "file for --evaluate, which prints the scores).")
    parser.add_argument("--format", type=str, default=None, choices=FORMATS,
                        help="Format of the file of --input or --evaluate (default: from the end of its name).")
    parser.add_argument("--split", type=str, default="test",
                        help="Split of a saved dataset that is read (default: test).")
    parser.add_argument("--words", action="store_true",
                        help="The sentences are already split into words, at their spaces: they are not split again.")
    args = parser.parse_args()

    if args.list:
        if args.texts or args.input or args.evaluate or args.model_dir:
            parser.error("--list only goes with --task and --model")
        return args
    if args.model_dir is None and (args.task is None or args.model is None):
        parser.error("give the model: --task and --model, or --model-dir (the runs are listed by --list)")
    if args.model_dir is not None and (args.model is not None or args.seed is not None):
        parser.error("--model-dir is the directory of one model: --model and --seed choose one among runs")
    if sum(bool(source) for source in (args.texts, args.input, args.evaluate)) > 1:
        parser.error("give texts, or --input, or --evaluate")
    return args


def asked_examples(model, words):
    """The examples that are typed, until an empty line or the end of the input."""
    columns = list(model.task.inputs)
    interactive = sys.stdin.isatty()
    if interactive:
        print(f"{model.name}: type {'a text' if len(columns) == 1 else 'the texts of an example'}, an empty line to stop.", file=sys.stderr)
    while True:
        texts = []
        for column in columns:
            try:
                texts.append(input((f"{column}> " if len(columns) > 1 else "> ") if interactive else ""))
            except EOFError:
                return
        if not any(text.strip() for text in texts):
            return
        yield model.example(*texts, words=words)


def typed_examples(model, args):
    """The examples of the texts of the command line, or else the ones that are asked for."""
    if not args.texts:
        return asked_examples(model, args.words)
    size = len(model.task.inputs)
    if len(args.texts) % size:
        raise PredictError(f"{model.name} reads {size} texts for an example ({', '.join(model.task.inputs)}): {len(args.texts)} given")
    return [model.example(*args.texts[first:first + size], words=args.words) for first in range(0, len(args.texts), size)]


def predict_file(model, args):
    """Write the predictions for the examples of a file."""
    read = read_examples(args.input, model, words=args.words, split=args.split, file_format=args.format)
    if not read.examples:
        raise PredictError(f"{args.input} has no example")
    predictions = model.predict(read.examples)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            write_predictions(f, model, read, predictions)
    else:
        write_predictions(sys.stdout, model, read, predictions)


def evaluate_file(model, args):
    """Print the scores of the task for the annotated examples of a file, and write the predictions in the file that is given for them."""
    read = read_examples(args.evaluate, model, annotated=True, words=args.words, split=args.split, file_format=args.format)
    if not read.examples:
        raise PredictError(f"{args.evaluate} has no example")
    references, identifiers = references_of(model, read)
    predictions = model.predict(read.examples)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            write_predictions(f, model, read, predictions, references)
    metrics = model.task.metrics(references, predictions, model.labels, identifiers)
    print(f"{len(predictions)} examples of {args.evaluate}")
    print(scores_as_text(metrics, model.labels))


def what_is_loaded(model):
    """A line that says which model predicts."""
    description = model.description
    name = description.get("model") or description["directory"]
    if os.path.isabs(name):
        # A model that is not one of the HuggingFace Hub: the name of its directory
        name = os.path.basename(os.path.normpath(name))
    line = f"Model: {name} fine-tuned on {model.name}"
    if description.get("seed") is not None:
        line += f", seed {description['seed']}"
    validation = description.get("validation") or {}
    if validation.get("score") is not None:
        line += f" (validation{' ' + validation['metric'] if validation.get('metric') else ''}: {validation['score']:.4f}"
        if description.get("chosen_among", 1) > 1:
            line += f", the best of the {description['chosen_among']} runs that kept their model"
        line += ")"
    return line


def main():
    args = arguments()
    if args.list:
        print(list_runs(args.task, args.model))
        return

    model = load(args.task, args.model, args.seed, args.model_dir)
    # On the standard error: the standard output only has the predictions
    print(what_is_loaded(model), file=sys.stderr)
    unnamed = len(model.labels) > 1 and all(re.fullmatch(r"LABEL_\d+", label) for label in model.labels)
    if unnamed:
        print("The names of the classes were not saved with this model, which an earlier version of the scripts kept: "
              "a class is shown as LABEL_<its number>.", file=sys.stderr)

    if args.input:
        predict_file(model, args)
        return
    if args.evaluate:
        evaluate_file(model, args)
        return
    for example in typed_examples(model, args):
        print(shown(model, example, model.predict([example])[0]), flush=True)


if __name__ == '__main__':
    try:
        main()
    except PredictError as error:
        sys.exit(str(error))
