"""Tests of what a recipe says in offline mode when the copy of its dataset or of its model is missing.

`load_from_disk` and `from_pretrained` only say that they do not find a directory: the
recipe has to say which script saves these copies.
"""

import argparse
import shlex

import pytest

from helpers import REPO, Task, run_command

DATASETS_SCRIPT = f"`python {REPO / 'download_datasets_locally.py'}`"
# The command that saves the model of the run
MODELS_SCRIPT = f"`python {REPO / 'download_models_locally.py'} --models Some-Org/Some-Model`"
HUB = "on a machine that has access to the HuggingFace Hub"


@pytest.fixture
def arguments(tmp_path):
    """The arguments of a run in offline mode, whose model is given by its name on the Hub."""
    def arguments(**changes):
        values = {
            "offline": True, "dataset": "DrBenchmark/CAS", "subset": "pos", "data_dir": f"{tmp_path}/data/",
            # `utils.parse_args` writes the directory of the copy of the model, and keeps its name
            "model_name": str(tmp_path / "models" / "some-org_some-model"), "hub_model_name": "Some-Org/Some-Model",
        }
        return argparse.Namespace(**{**values, **changes})

    return arguments


@pytest.fixture
def saved_dataset(tmp_path):
    (tmp_path / "data" / "local_hf_pos").mkdir(parents=True)


@pytest.fixture
def saved_model(tmp_path):
    (tmp_path / "models" / "some-org_some-model").mkdir(parents=True)


def message_of(training, *args):
    with pytest.raises(SystemExit) as stopped:
        training.check_local_copies(*args)
    # A text, which Python writes on the standard error before it exits with the code 1
    assert isinstance(stopped.value.code, str)
    return stopped.value.code


def test_nothing_is_said_when_the_copies_exist(training, arguments, saved_dataset, saved_model):
    assert training.check_local_copies(arguments()) is None


def test_missing_dataset(training, arguments, saved_model, tmp_path):
    assert message_of(training, arguments()) == (
        f"The dataset DrBenchmark/CAS (subset pos) is not in {tmp_path}/data/local_hf_pos, where it is read in offline mode.\n"
        f"Save the datasets with {DATASETS_SCRIPT}, {HUB}.\n"
        "\n"
        "To read the dataset from the HuggingFace Hub instead, give `--offline false` "
        f"or set `offline: False` in {REPO / 'config.yaml'}."
    )


def test_missing_dataset_without_subset(training, arguments, saved_model, tmp_path):
    message = message_of(training, arguments(dataset="DrBenchmark/DiaMED", subset=None))
    assert message.startswith(f"The dataset DrBenchmark/DiaMED is not in {tmp_path}/data/local_hf_None, where")


def test_missing_model(training, arguments, saved_dataset, tmp_path):
    assert message_of(training, arguments()) == (
        f"The model Some-Org/Some-Model is not a directory, and its copy is not in {tmp_path}/models/some-org_some-model, "
        "where it is read in offline mode.\n"
        f"Save it with {MODELS_SCRIPT}, {HUB}.\n"
        "\n"
        "To read the model from the HuggingFace Hub instead, give `--offline false` "
        f"or set `offline: False` in {REPO / 'config.yaml'}."
    )


def test_missing_dataset_and_model(training, arguments):
    message = message_of(training, arguments())
    assert DATASETS_SCRIPT in message and MODELS_SCRIPT in message
    assert "To read the dataset and the model from the HuggingFace Hub instead" in message


def test_model_given_by_a_path_is_not_looked_for_in_the_copies(training, arguments, saved_dataset, tmp_path):
    """`utils.parse_args` only keeps the name of a model that is not a path which exists."""
    assert training.check_local_copies(arguments(model_name=str(tmp_path), hub_model_name=None)) is None


def test_missing_dataset_which_is_not_on_the_hub(training, arguments, saved_model, tmp_path):
    """DEFT-2019: `download_datasets_locally.py` does not save it, and it cannot be read from the Hub."""
    message = message_of(training, arguments(dataset="DrBenchmark/DEFT2019", subset=None), True)
    assert message == (
        f"The dataset DrBenchmark/DEFT2019 is not in {tmp_path}/data/local_hf_None, where it is read in offline mode.\n"
        "It is not on the HuggingFace Hub and `download_datasets_locally.py` does not save it: see the README of its recipe."
    )


@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "frenchmedmcqa-mcqa", "deft2019-ner"])
def test_recipe_stops_with_what_to_do(task_name, tmp_path):
    """Every training script checks the copies before it loads anything, the one that does not use a shared module too."""
    task = Task(task_name)
    options = ["--offline", "true", "--seed", "0", "--data_dir", f"{tmp_path}/data/", "--run_dir", f"{tmp_path}/"]
    command = task.command("Some-Org/Some-Model") + " " + " ".join(shlex.quote(option) for option in options)
    completed = run_command(command, cwd=task.scripts_dir)

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert f"The dataset {task.config['dataset']}" in completed.stderr
    assert f"is not in {tmp_path}/data/local_hf_{task.subset}, where it is read in offline mode." in completed.stderr
    if task_name == "deft2019-ner":
        assert "see the README of its recipe" in completed.stderr and DATASETS_SCRIPT not in completed.stderr
    else:
        assert DATASETS_SCRIPT in completed.stderr
    # `models/` at the root of the repository, where `download_models_locally.py` saves the models
    assert f"its copy is not in {REPO / 'models' / 'some-org_some-model'}, where" in completed.stderr
    assert MODELS_SCRIPT in completed.stderr
