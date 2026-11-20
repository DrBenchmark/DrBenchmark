"""Consistency between `run.py`, the recipes, the scripts for SLURM and the documentation.

These tests do not train anything: they check that the files that describe the tasks
agree with each other.
"""

import os
import re

import pytest

from helpers import ALL_TASKS, RECIPE_NAMES, RECIPES, REPO, TASKS, load_module


def task_ids(task):
    return task.name


def test_every_recipe_has_a_task():
    assert sorted({task.recipe for task in ALL_TASKS}) == RECIPE_NAMES


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_launcher(task):
    assert task.launcher_path.is_file()
    assert os.access(task.launcher_path, os.X_OK), f"{task.launcher_path} is not executable"
    assert (task.scripts_dir / task.script).is_file()
    assert task.config_path.is_file()


@pytest.mark.parametrize("recipe", RECIPE_NAMES)
def test_no_unused_file(recipe):
    """Every launcher, training script and YAML file of a recipe belongs to a task of `run.py`."""
    tasks = [task for task in ALL_TASKS if task.recipe == recipe]
    scripts_dir = RECIPES / recipe / "scripts"

    assert sorted(path.name for path in scripts_dir.glob("run*.sh")) == sorted({task.launcher for task in tasks})
    training_scripts = sorted(path.name for path in scripts_dir.glob("*.py"))
    assert training_scripts == sorted({task.script for task in tasks})
    assert sorted((RECIPES / recipe / "yaml").glob("*.yaml")) == sorted({task.config_path for task in tasks})


@pytest.mark.parametrize("recipe", RECIPE_NAMES)
def test_common_package_is_shared(recipe):
    """Every recipe uses the code of `recipes/common/`, through a symbolic link."""
    link = RECIPES / recipe / "scripts" / "common"
    assert link.is_symlink()
    assert link.resolve() == (RECIPES / "common").resolve()


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_training_script_uses_the_shared_code(task):
    """A training script only states what is specific to its task: the rest is in `recipes/common/`."""
    script = (task.scripts_dir / task.script).read_text()
    assert re.search(r"^from common import ", script, re.M)
    for duplicated in ("TrainingArguments(", "Trainer(", "load_dataset(", "parse_args("):
        assert duplicated not in script, f"{duplicated} should be called by recipes/common/"


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_yaml_file(task):
    config = task.config
    required = {
        "epochs", "model_name", "batch_size", "weight_decay", "learning_rate",
        "output_dir", "run_dir", "data_dir", "subset", "fewshot", "task", "dataset",
    }
    assert required <= set(config), required - set(config)

    assert config["task"].split("-")[0] == task.recipe
    # `utils.parse_args` appends the subset to the name: e3c-ner -> e3c-ner_clinical
    assert task.name.startswith(config["task"])
    assert config["dataset"].startswith("DrBenchmark/")
    assert config["dataset"].split("/")[1].lower() == task.recipe
    assert config["fewshot"] == 1.0
    assert (config["output_dir"], config["run_dir"], config["data_dir"]) == ("../save_models/", "../runs/", "../data/")


def test_datasets_saved_for_the_offline_mode():
    """`download_datasets_locally.py` saves every dataset the recipes read in offline mode.

    DEFT-2019 is the exception: its access is restricted and it is not on the Hub.
    """
    module = load_module("download_datasets_locally", REPO / "download_datasets_locally.py")
    saved = {(corpus.lower(), subset) for corpus, subset in module.ds}
    read = {(task.recipe, task.subset) for task in ALL_TASKS}
    assert read - saved == {("deft2019", None)}
    assert saved <= read

    datasets = {task.recipe: task.config["dataset"] for task in ALL_TASKS}
    for corpus, _ in module.ds:
        assert f"DrBenchmark/{corpus}" == datasets[corpus.lower()]


@pytest.mark.parametrize("script", ["run_all_slurm.sh", "run_all_jean_zay.sh"])
def test_slurm_script_runs_every_task(script):
    jobs = re.findall(
        r"""^\s*"cd \./recipes/(\w+)/scripts/ && srun bash (\S+) '\$MODEL_NAME'(?: '(\w+)')?"$""",
        (REPO / script).read_text(),
        re.M,
    )
    commands = sorted(
        (recipe, f"./{launcher} {{model_name}}" + (f" {subset}" if subset else ""))
        for recipe, launcher, subset in jobs
    )
    assert commands == sorted((task.recipe, TASKS[task.name]) for task in ALL_TASKS)


@pytest.mark.parametrize("script", ["run_all_slurm.sh", "run_all_jean_zay.sh"])
def test_slurm_script_does_the_runs_of_the_benchmark(script):
    """One job per run of each task for each model: when the job array is too small, runs are silently skipped."""
    source = (REPO / script).read_text()
    runs = int(re.search(r"^NBR_RUNS=(\d+)$", source, re.M).group(1))
    assert runs == load_module("drbenchmark_number_of_runs", REPO / "tools" / "number_of_runs.py").NB_RUNS

    # The script takes every line of `models.txt` as a model
    models = (REPO / "models.txt").read_text().splitlines()
    first, last = re.search(r"^#SBATCH --array=(\d+)-(\d+)%\d+", source, re.M).groups()
    assert (int(first), int(last)) == (0, len(ALL_TASKS) * len(models) * runs - 1)


def test_run_one_jean_zay_lists_every_task():
    """`run_one_jean_zay.sh` proposes one line per task, with the names `run.py` accepts."""
    script = (REPO / "run_one_jean_zay.sh").read_text()
    proposed = re.findall(r"^#? ?TASK='([^']+)'$", script, re.M)
    assert sorted(proposed) == sorted(TASKS)
    # Exactly one of them is not commented out
    assert len(re.findall(r"^TASK='([^']+)'$", script, re.M)) == 1


def test_recipes_readme_lists_every_task():
    """The table of `recipes/README.md` gives the name and the command of each task."""
    rows = re.findall(r"\| `([a-z0-9]+-[a-z_]+)`\s*\| `(\./run[^`]*)`\s*\|$", (RECIPES / "README.md").read_text(), re.M)
    documented = {name: command.replace("<model>", "{model_name}") for name, command in rows}
    assert documented == TASKS


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_recipe_readme_documents_the_task(task):
    readme = (RECIPES / task.recipe / "README.md").read_text()
    assert task.name in readme
    assert f"`{task.launcher}`" in readme
    assert f"`yaml/{task.config_path.name}`" in readme
    if task.subset:
        assert f"`{task.subset}`" in readme
