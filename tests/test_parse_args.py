"""Tests of `recipes/utils.py`: command line parsing and naming of the runs."""

import os
import re
import sys

import pytest
import yaml

from helpers import ALL_TASKS, REPO, Task

MODEL = "Dr-BERT/DrBERT-7GB"
GLOBAL_CONFIG = yaml.safe_load((REPO / "config.yaml").read_text())


def task_ids(task):
    return task.name


@pytest.fixture
def parse(utils, monkeypatch):
    """Parse the arguments as the script of a task does.

    The script is run from its `scripts/` directory with the arguments given by its
    launcher, followed by `extra_args`.
    """
    def parse(task, *extra_args, model=MODEL):
        if isinstance(task, str):
            task = Task(task)
        monkeypatch.chdir(task.scripts_dir)
        monkeypatch.setattr(sys, "argv", [task.script, *task.script_args(model), *extra_args])
        return utils.parse_args()

    return parse


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_task_name_and_subset(task, parse):
    """The name used in `run.py` is the one written in the results of the run."""
    args = parse(task)
    assert args.task == task.name
    assert args.subset == task.subset
    assert args.dataset == task.config["dataset"]


def test_yaml_file_gives_the_default_values(parse):
    task = Task("cas-pos")
    args = parse(task)
    for name in ("epochs", "batch_size", "weight_decay", "learning_rate", "fewshot", "data_dir", "run_dir"):
        assert getattr(args, name) == task.config[name], name
    assert args.output_dir == task.config["output_dir"].rstrip("/")


def test_global_config_is_applied(parse):
    args = parse("cas-pos")
    for name, value in GLOBAL_CONFIG.items():
        assert getattr(args, name) == value, name


def test_command_line_overrides_the_yaml_file(parse):
    args = parse(
        "cas-pos",
        "--epochs", "3", "--batch_size", "2", "--learning_rate", "0.5", "--weight_decay", "0.25",
        "--seed", "7", "--fewshot", "0.5", "--max_position_embeddings", "128", "--run_dir", "/tmp/somewhere/",
    )
    assert (args.epochs, args.batch_size, args.learning_rate, args.weight_decay) == (3, 2, 0.5, 0.25)
    assert (args.seed, args.fewshot, args.max_position_embeddings) == (7, 0.5, 128)
    assert args.run_dir == "/tmp/somewhere/"


def test_run_name_and_paths(parse, utils):
    args = parse("e3c-ner_clinical", "--seed", "1")
    name = utils.create_run_name(args)
    assert re.fullmatch(r"DrBenchmark-e3c-ner_clinical-[0-9a-f]{16}", name)
    assert utils.get_run_path(args) == os.path.join(args.run_dir, f"{name}.json")
    assert utils.get_model_path(args) == os.path.join(args.output_dir, name)


def test_run_name_depends_on_the_arguments(parse, utils):
    reference = utils.create_run_name(parse("cas-pos", "--seed", "1"))
    assert utils.create_run_name(parse("cas-pos", "--seed", "1")) == reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "2")) != reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "1", "--epochs", "3")) != reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "1", model="other/model")) != reference


def test_seed_is_drawn_when_none_is_given(parse):
    seed = parse("cas-pos").seed
    assert isinstance(seed, int) and 0 <= seed < 2**32


def test_run_already_computed_is_skipped(parse, utils, tmp_path):
    arguments = ("--seed", "1", "--run_dir", f"{tmp_path}/")
    run_path = utils.get_run_path(parse("cas-pos", *arguments))
    with open(run_path, "w") as f:
        f.write("{}")

    with pytest.raises(SystemExit) as exit_info:
        parse("cas-pos", *arguments)
    assert exit_info.value.code == 0
    # Without a seed, another one is drawn instead of skipping the run
    assert parse("cas-pos", "--run_dir", f"{tmp_path}/").seed is not None


def test_existing_model_path_is_kept(parse, tmp_path):
    assert parse("cas-pos", model=str(tmp_path)).model_name == str(tmp_path)


def test_model_path_relative_to_the_repository(parse):
    """`run.py` is run from the root of the repository, the scripts from `scripts/`."""
    assert parse("cas-pos", model="assets").model_name == os.path.join("..", "..", "..", "assets")


def test_hub_model_in_offline_mode(parse):
    args = parse("cas-pos", "--offline", "True")
    assert args.offline is True
    assert args.model_name == os.path.join("..", "..", "..", "models", "dr-bert_drbert-7gb")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_fewshot_and_max_train_samples_are_exclusive(parse):
    with pytest.raises(ValueError):
        parse("cas-pos", "--fewshot", "0.5", "--max_train_samples", "10")
    assert parse("cas-pos", "--max_train_samples", "10").max_train_samples == 10


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known bug: the default value of --gradient_accumulation_steps (1) overrides the YAML file.",
)
@pytest.mark.parametrize(
    "task", [task for task in ALL_TASKS if "gradient_accumulation_steps" in task.config], ids=task_ids,
)
def test_gradient_accumulation_of_the_yaml_file_is_kept(task, parse):
    assert parse(task).gradient_accumulation_steps == task.config["gradient_accumulation_steps"]


@pytest.mark.parametrize("value, expected", [
    ("True", True), ("true", True), ("yes", True), ("1", True),
    ("False", False), ("false", False), ("no", False), ("0", False),
])
def test_offline_mode_on_the_command_line(value, expected, parse, monkeypatch):
    """`--offline` overrides `config.yaml`, in both directions."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    args = parse("cas-pos", "--offline", value)
    assert args.offline is expected
    assert ("HF_HUB_OFFLINE" in os.environ) is expected
    if not expected:
        # The model is downloaded from the Hub: its name is kept as it is
        assert args.model_name == MODEL


def test_offline_mode_needs_a_boolean(parse, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("cas-pos", "--offline", "maybe")
    assert exit_info.value.code == 2
    assert "expected true or false" in capsys.readouterr().err
