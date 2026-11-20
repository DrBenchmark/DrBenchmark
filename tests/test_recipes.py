"""End-to-end tests: each task of `run.py` is trained and evaluated on synthetic data.

The launchers are called exactly as `run.py` calls them, with a tiny model and one
epoch. The scores are meaningless: the tests only check that every recipe runs and
writes the results the tools of `tools/` expect.
"""

import importlib.metadata
import math
import os
import platform
from pathlib import Path

import pytest

import synthetic
from helpers import ALL_TASKS, RECIPES, Task, read_json, run_command

pytestmark = pytest.mark.e2e

N_TEST = synthetic.SPLIT_SIZES["test"]


def task_ids(task):
    return task.name


def assert_score(value):
    assert isinstance(value, float) and 0.0 <= value <= 1.0, value


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_recipe(task, run_recipe, tiny_bert):
    run = run_recipe(task.name, tiny_bert)
    result = run.result

    assert run.run_files[0].name.startswith(f"DrBenchmark-{task.name}-")
    assert result["hyperparameters"]["task"] == task.name
    assert result["hyperparameters"]["subset"] == task.subset
    assert result["trainer_state"]["epoch"] == 1

    # The best model is saved next to the results
    assert (Path(result["model_name"]) / "config.json").is_file()

    # The versions that computed the run
    environment = result["environment"]
    assert environment["python"] == platform.python_version()
    for library in ("torch", "transformers", "datasets", "numpy"):
        assert environment[library] == importlib.metadata.version(library)
    assert {"commit", "modified"} <= set(environment)

    # One prediction per example of the test set
    predictions = result["predictions"]
    assert len(predictions["identifiers"]) == N_TEST
    assert len(predictions["real_labels"]) == N_TEST
    assert len(predictions["system_predictions"]) == N_TEST

    # The metrics read by `tools/CollectAndAverageScores.py`
    metrics = result["metrics"]
    if task.kind in ("pos", "ner"):
        for name in ("overall_precision", "overall_recall", "overall_f1", "overall_accuracy"):
            assert_score(metrics[name])
        # One predicted label per word
        for reference, prediction in zip(predictions["real_labels"], predictions["system_predictions"]):
            assert len(reference) == len(prediction)
    elif task.kind == "cls":
        assert_score(metrics["weighted avg"]["f1-score"])
        assert_score(metrics["macro avg"]["f1-score"])
    elif task.kind == "mcqa":
        assert_score(metrics["hamming_score"])
        assert_score(metrics["exact_match"])
    elif task.kind == "regr":
        assert_score(metrics["EDRM"])
        # The correlation is undefined (NaN) when the tiny model predicts a constant
        assert isinstance(metrics["spearman_correlation_coef"], float)
    else:
        pytest.fail(f"Unknown kind of task: {task.kind}")


def names(directory):
    return sorted(path.name for path in directory.iterdir() if path.name != "__pycache__")


# A launcher without subset, and one with a subset
@pytest.mark.parametrize("task_name", ["cas-pos", "e3c-ner_clinical"])
def test_recipe_started_from_another_directory(task_name, run_recipe, tiny_bert, synthetic_data, tmp_path):
    """The launcher is run from a directory out of the repository, with paths relative to that directory."""
    task = Task(task_name)
    os.symlink(tiny_bert, tmp_path / "model")
    os.symlink(synthetic_data / task.recipe, tmp_path / "data")
    (tmp_path / "runs").mkdir()
    recipe = RECIPES / task.recipe
    before = {name: names(recipe / name) for name in ("runs", "save_models", "scripts")}

    arguments = "--data_dir data/ --run_dir runs/ --output_dir models/ --offline True --seed 0 --epochs 1"
    completed = run_command(f"{task.command('model', cwd=tmp_path)} {arguments}", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert f"Working directory: {task.scripts_dir}" in completed.stderr

    # The results are written in the directory given, and nothing in the recipe
    (run_file,) = (tmp_path / "runs").glob("*.json")
    result = read_json(run_file)
    assert {name: names(recipe / name) for name in before} == before

    # The paths are written from `scripts/`, whatever the directory the launcher is run from
    hyperparameters = result["hyperparameters"]
    assert hyperparameters["config"] == task.config_arg
    for argument, typed in (("model_name", "model"), ("data_dir", "data"), ("run_dir", "runs"), ("output_dir", "models")):
        path = hyperparameters[argument]
        assert not os.path.isabs(path), argument
        assert os.path.realpath(task.scripts_dir / path) == os.path.realpath(tmp_path / typed), argument

    best_model = task.scripts_dir / result["model_name"]
    assert os.path.realpath(best_model).startswith(str(tmp_path / "models"))
    assert (best_model / "config.json").is_file()

    # The run is the one the launcher does from `scripts/`
    reference = run_recipe(task_name, tiny_bert).result
    assert result["predictions"]["identifiers"] == reference["predictions"]["identifiers"]
    assert result["predictions"]["real_labels"] == reference["predictions"]["real_labels"]
    assert result["trainer_state"]["global_step"] == reference["trainer_state"]["global_step"]
    assert set(result["metrics"]) == set(reference["metrics"])


def test_flaubert_sequence_classification(run_recipe, tiny_flaubert):
    """The tokenizer of FlauBERT is a slow one: check that the recipes accept it."""
    assert_score(run_recipe("cas-cls", tiny_flaubert).result["metrics"]["weighted avg"]["f1-score"])


@pytest.mark.parametrize("task_name", ["cas-pos", "quaero-ner_emea"])
def test_flaubert_token_classification(task_name, run_recipe, tiny_flaubert):
    run = run_recipe(task_name, tiny_flaubert)
    if run.returncode != 0 and "inconsistent numbers of samples" in run.log:
        pytest.xfail(
            "Known bug: with FlauBERT, the scripts predict one label per sub-token but "
            "compare them with one reference label per word, which seqeval rejects."
        )
    assert_score(run.result["metrics"]["overall_f1"])


GRADIENT_ACCUMULATION_TASKS = [task for task in ALL_TASKS if "gradient_accumulation_steps" in task.config]


@pytest.mark.parametrize("task", GRADIENT_ACCUMULATION_TASKS, ids=task_ids)
def test_gradient_accumulation_is_applied(task, run_recipe, tiny_bert):
    """The weights are updated once per effective batch: `batch_size * gradient_accumulation_steps` examples."""
    config = task.config
    examples_per_update = config["batch_size"] * config["gradient_accumulation_steps"]
    expected_updates = math.ceil(synthetic.SPLIT_SIZES["train"] / examples_per_update)
    # Without accumulation, there would be one update per batch
    assert expected_updates < math.ceil(synthetic.SPLIT_SIZES["train"] / config["batch_size"])

    result = run_recipe(task.name, tiny_bert).result
    assert result["hyperparameters"]["gradient_accumulation_steps"] == config["gradient_accumulation_steps"]
    assert result["trainer_state"]["global_step"] == expected_updates


def test_gradient_accumulation_given_on_the_command_line(run_recipe, tiny_bert):
    """6 batches of 2 examples, accumulated 3 by 3: the weights are updated twice instead of 6 times."""
    assert synthetic.SPLIT_SIZES["train"] == 12
    run = run_recipe("cas-cls", tiny_bert, "--batch_size", "2", "--gradient_accumulation_steps", "3")
    assert run.result["trainer_state"]["global_step"] == 2


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known bug: `metric_for_best_model='rmse'` is used without `greater_is_better=False`, "
           "so the checkpoint with the highest validation RMSE is kept.",
)
@pytest.mark.parametrize("task", [task for task in ALL_TASKS if task.kind == "regr"], ids=task_ids)
def test_regression_keeps_the_checkpoint_with_the_lowest_rmse(task, run_recipe, tiny_bert):
    # A high learning rate makes the RMSE clearly different from one epoch to the next
    run = run_recipe(task.name, tiny_bert, "--epochs", "3", "--learning_rate", "0.01")
    state = run.result["trainer_state"]
    rmse = [log["eval_rmse"] for log in state["log_history"] if "eval_rmse" in log]
    assert len(rmse) >= 3
    assert state["best_metric"] == pytest.approx(min(rmse))


def expected_labels(task):
    """The names of the outputs of a model of a task: the classes of the column of its dataset that the model learns."""
    features = synthetic.dataset_of(task.recipe, task.subset)["train"].features
    if task.kind in ("pos", "ner"):
        return features["pos_tags" if task.kind == "pos" else "ner_tags"].feature.names
    if task.kind == "regr":
        return ["LABEL_0"]
    for column in ("label", "correct_cible", "icd-10", "number_correct_answers"):
        if column in features:
            return features[column].names
    return features["specialities"].feature.names


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_names_of_the_classes_are_saved_with_the_model(task, run_recipe, tiny_bert):
    """Without them a prediction is a number, which only the dataset explains."""
    config = read_json(os.path.join(run_recipe(task.name, tiny_bert).result["model_name"], "config.json"))
    names = [config["id2label"][str(index)] for index in range(len(config["id2label"]))]
    if task.kind == "mcqa":
        # Every set of answers among five
        assert len(names) == 31 and sorted(names, key=lambda name: (len(name), name))[:6] == ["a", "b", "c", "d", "e", "ab"]
    else:
        assert names == expected_labels(task)
    assert config["label2id"] == {name: index for index, name in enumerate(names)}


def test_kept_model_is_described(run_recipe, tiny_bert, training):
    """What a run writes next to its model: the directory is enough to know what the model is."""
    result = run_recipe("quaero-ner_emea", tiny_bert).result
    directory = result["model_name"]
    description = read_json(os.path.join(directory, training.DESCRIPTION))
    assert training.DESCRIPTION == "drbenchmark.json"

    hyperparameters = result["hyperparameters"]
    assert description == {
        "task": "quaero-ner_emea",
        "kind": "token classification",
        "dataset": "DrBenchmark/QUAERO",
        "subset": "emea",
        "model": str(tiny_bert),
        "seed": 0,
        "script": "recipes/quaero/scripts/finetuning_bert_ner.py",
        "inputs": {"tokens": "words"},
        "labels": synthetic.UMLS_TAGS,
        "max_length": hyperparameters["max_position_embeddings"],
        "hyperparameters": {
            name: hyperparameters[name]
            for name in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay", "fewshot")
        },
        "validation": {"metric": "accuracy", "score": result["trainer_state"]["best_metric"]},
        "metrics": result["metrics"],
        "environment": result["environment"],
    }
    # The results say which score chose the checkpoint too
    assert result["validation"] == description["validation"]

    card = open(os.path.join(directory, "README.md"), encoding="utf-8").read()
    header, text = card.split("---\n")[1:3]
    assert header.splitlines() == [
        "language: fr", "library_name: transformers", "tags:", "- drbenchmark", "- biomedical",
        "pipeline_tag: token-classification", "datasets:", "- DrBenchmark/QUAERO",
    ]
    assert text.startswith("\n# tiny-bert fine-tuned on quaero-ner_emea\n")
    for expected in (
        "token classification, on the dataset `DrBenchmark/QUAERO` (subset `emea`)", "with the seed 0", "the best `accuracy`",
        f"| overall_f1 | {result['metrics']['overall_f1']:.4f} |", "- learning_rate: ", "- at most 512 sub-word tokens per example",
        "The model reads `tokens` (the words of a sentence).", "`recipes/quaero/scripts/finetuning_bert_ner.py`",
        'python predict.py --model-dir <this directory> "..."',
    ):
        assert expected in text, expected


@pytest.mark.parametrize("task_name, pipeline, dataset", [
    ("cas-cls", "text-classification", ["datasets:", "- DrBenchmark/CAS"]),
    # A dataset that is not on the HuggingFace Hub
    ("deft2019-ner", "token-classification", []),
    ("clister-regr", "text-classification", ["datasets:", "- DrBenchmark/CLISTER"]),
])
def test_card_of_a_kept_model(task_name, pipeline, dataset, run_recipe, tiny_bert):
    directory = run_recipe(task_name, tiny_bert).result["model_name"]
    header = open(os.path.join(directory, "README.md"), encoding="utf-8").read().split("---\n")[1].splitlines()
    assert header[5:] == [f"pipeline_tag: {pipeline}", *dataset]


def test_model_is_not_kept_when_it_is_asked(run_recipe, tiny_bert):
    """`--keep_model false`: the run is the same one, with its results, without its model."""
    kept = run_recipe("cas-pos", tiny_bert)
    not_kept = run_recipe("cas-pos", tiny_bert, "--keep_model", "false")
    result = not_kept.result

    assert result["model_name"] is None
    assert not (not_kept.directory / "save_models").exists() or list((not_kept.directory / "save_models").iterdir()) == []
    assert (result["hyperparameters"]["keep_model"], kept.result["hyperparameters"]["keep_model"]) == (False, True)
    # What the run computed does not depend on what is kept of it
    assert result["metrics"] == kept.result["metrics"] and result["predictions"] == kept.result["predictions"]
    assert os.path.isdir(kept.result["model_name"])
