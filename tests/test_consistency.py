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


def test_code_of_the_recipes_has_no_symbolic_link():
    """A symbolic link may be lost on Windows, or when the repository is extracted from a zip archive.

    The scripts of the recipes used to import the code they share through a link `scripts/common`.
    """
    directories = [RECIPES / "common"] + [RECIPES / recipe / name for recipe in RECIPE_NAMES for name in ("scripts", "yaml")]
    assert [path for directory in directories for path in directory.rglob("*") if path.is_symlink()] == []
    assert not any((RECIPES / recipe / "scripts" / "common").exists() for recipe in RECIPE_NAMES)


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_training_script_finds_the_shared_code_by_itself(task):
    """A training script adds `recipes/` to the search path, from its own location, before it imports `common`."""
    lines = (task.scripts_dir / task.script).read_text().splitlines()
    path = lines.index("sys.path.insert(0, str(Path(__file__).resolve().parents[2]))")
    assert (task.scripts_dir / task.script).resolve().parents[2] == RECIPES.resolve()
    imports = [number for number, line in enumerate(lines) if line.startswith(("from common", "import common"))]
    assert imports and path < min(imports)


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


def names_of_tasks(text):
    """The names of tasks of `run.py` that are written in a text."""
    return re.findall("|".join(sorted(TASKS, key=len, reverse=True)), text)


@pytest.mark.parametrize("script", ["run_all_slurm.sh", "run_all_jean_zay.sh"])
def test_slurm_script_asks_run_py_for_every_task(script):
    """The tasks are only listed in `run.py`: the scripts for SLURM ask it for a run of all of them."""
    source = (REPO / script).read_text()
    assert 'srun python run.py --tasks all --nb-run "$NBR_RUNS" --job "$SLURM_ARRAY_TASK_ID"' in source.splitlines()
    # Neither a task nor a launcher is written in the script
    assert names_of_tasks(source) == []
    assert "run_task_" not in source and "run.sh" not in source


@pytest.mark.parametrize("script", ["run_all_slurm.sh", "run_all_jean_zay.sh"])
def test_slurm_script_does_the_runs_of_the_benchmark(script):
    """One job per run of each task for each model: when the job array is too small, runs are silently skipped."""
    source = (REPO / script).read_text()
    runs = int(re.search(r"^NBR_RUNS=(\d+)$", source, re.M).group(1))
    assert runs == load_module("drbenchmark_number_of_runs", REPO / "tools" / "number_of_runs.py").NB_RUNS

    # The models of `models.txt`: its lines, but the empty ones and the comments
    lines = [line.strip() for line in (REPO / "models.txt").read_text().splitlines()]
    models = [line for line in lines if line and not line.startswith("#")]
    first, last = re.search(r"^#SBATCH --array=(\d+)-(\d+)%\d+", source, re.M).groups()
    assert (int(first), int(last)) == (0, len(ALL_TASKS) * len(models) * runs - 1)


def test_task_of_run_one_jean_zay_is_one_of_run_py():
    """`run_one_jean_zay.sh` does one task, which has a name that `run.py` accepts."""
    script = (REPO / "run_one_jean_zay.sh").read_text()
    (task,) = re.findall(r"^TASK='([^']+)'$", script, re.M)
    assert task in TASKS
    # The other tasks are not listed: their names are the ones of `python run.py --help`
    assert names_of_tasks(script) == [task]


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
