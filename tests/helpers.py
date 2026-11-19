"""Helpers shared by the tests: paths of the repository and description of the tasks."""

import importlib.util
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "recipes"


def load_module(name, path):
    """Import a Python file of the repository without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# {task name: command run from `recipes/<recipe>/scripts/`}, e.g.
# {"e3c-ner_clinical": "./run.sh {model_name} French_clinical"}
TASKS = load_module("drbenchmark_run", REPO / "run.py").task2script

RECIPE_NAMES = sorted(path.name for path in RECIPES.iterdir() if path.is_dir())


class Task:
    """Everything `run.py` and the launcher of a task say about it."""

    def __init__(self, name):
        self.name = name
        self.recipe, detail = name.split("-")
        # pos, cls, ner, regr or mcqa
        self.kind = detail.split("_")[0]
        launcher, model, *subset = TASKS[name].split()
        assert model == "{model_name}", TASKS[name]
        self.launcher = launcher[2:] if launcher.startswith("./") else launcher
        self.cli_subset = subset[0] if subset else None
        self.scripts_dir = RECIPES / self.recipe / "scripts"

    def __repr__(self):
        return f"Task({self.name})"

    @property
    def launcher_path(self):
        return self.scripts_dir / self.launcher

    @property
    def script(self):
        """Python script called by the launcher."""
        return re.search(r"^python (\S+\.py)", self.launcher_path.read_text(), re.M).group(1)

    @property
    def config_arg(self):
        """Path of the YAML file as the launcher gives it, relative to `scripts/`."""
        return re.search(r'--config="([^"]+)"', self.launcher_path.read_text()).group(1)

    @property
    def config_path(self):
        return (self.scripts_dir / self.config_arg).resolve()

    @property
    def config(self):
        return yaml.safe_load(self.config_path.read_text())

    @property
    def subset(self):
        """Subset of the dataset: given on the command line, or else by the YAML file."""
        return self.cli_subset or self.config["subset"]

    def script_args(self, model):
        """Arguments the launcher gives to the Python script."""
        args = [f"--config={self.config_arg}", f"--model={model}"]
        if self.cli_subset:
            args.append(f"--subset={self.cli_subset}")
        return args


ALL_TASKS = [Task(name) for name in TASKS]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
