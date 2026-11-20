"""Helpers shared by the tests: paths of the repository and description of the tasks."""

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RECIPES = REPO / "recipes"


def relative_path(path, start):
    """`path` as it is typed from the directory `start`.

    Symbolic links are resolved first: `..` is not the parent of a link, but of its target.
    """
    return os.path.relpath(os.path.realpath(path), os.path.realpath(start))


def load_module(name, path):
    """Import a Python file of the repository without running its `__main__` block."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# {task name: command run from `recipes/<recipe>/scripts/`}, e.g.
# {"e3c-ner_clinical": "./run.sh {model_name} French_clinical"}
TASKS = load_module("drbenchmark_run", REPO / "run.py").task2script

# The directories of `recipes/` with scripts: every one but `common/`, the code they share
RECIPE_NAMES = sorted(path.name for path in RECIPES.iterdir() if (path / "scripts").is_dir())

# A `models.txt` with what is not a model: empty lines and comments, some after spaces. Its
# last line does not end with a new line. Whatever reads the file has to find the same models.
MODELS_FILE = "# The models to evaluate\nfirst/model\n\n   # not/this-one\n\t\n#nor/this-one\n  second/model  "
MODELS_OF_THE_FILE = ["first/model", "second/model"]


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
        """Python script called by the launcher, which looks for it in its own directory."""
        return re.search(r'^python "\$scripts_dir/(\S+\.py)"', self.launcher_path.read_text(), re.M).group(1)

    @property
    def config_arg(self):
        """Path of the YAML file given by the launcher, relative to `scripts/`."""
        return re.search(r'--config="\$scripts_dir/([^"]+)"', self.launcher_path.read_text()).group(1)

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

    def script_args(self, model, cwd=None):
        """Arguments the launcher gives to the Python script.

        The path of the YAML file is given from `cwd`, the directory the launcher is run
        from: `scripts/` by default.
        """
        config = self.config_arg if cwd is None else relative_path(self.config_path, cwd)
        args = [f"--config={config}", f"--model={model}"]
        if self.cli_subset:
            args.append(f"--subset={self.cli_subset}")
        return args

    def command(self, model, cwd=None):
        """Command of `run.py` for the task, written to be run from `cwd`: `scripts/` by default."""
        command = TASKS[self.name].format(model_name=shlex.quote(str(model)))
        if cwd is not None:
            # ./run.sh <model> -> <path of scripts/ from cwd>/run.sh <model>
            assert command.startswith("./")
            command = shlex.quote(relative_path(self.scripts_dir, cwd)) + command[1:]
        return command


ALL_TASKS = [Task(name) for name in TASKS]


def run_command(command, cwd, timeout=900):
    """Run the command of a recipe from the directory `cwd`."""
    environ = dict(os.environ, TOKENIZERS_PARALLELISM="false")
    # The launchers call `python`: make it the interpreter running the tests
    environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + environ["PATH"]
    return subprocess.run(command, shell=True, cwd=cwd, env=environ, capture_output=True, text=True, timeout=timeout)


def copy_tools(repository):
    """Make `repository` a directory where the tools work as in the real one.

    The tools find `stats/`, `recipes/` and `models.txt` from their own location: their
    copies use the ones of `repository`.
    """
    shutil.copytree(REPO / "tools", repository / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    # `tools/WriteSlurmScriptForMissingRuns.py` reads the tasks in `run.py`
    shutil.copy(REPO / "run.py", repository)
    (repository / "stats").mkdir()


def run_tool(script, repository, *arguments, cwd):
    """Run the copy of a tool which is in `repository`, from the directory `cwd`."""
    return subprocess.run(
        [sys.executable, str(repository / "tools" / script), *arguments],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )


def run_without_slurm(script, cwd, directory, task_id=0):
    """Run a script for SLURM with bash, from the directory `cwd`, as the job `task_id` of its array.

    The programs the script calls (`srun`, `conda`, `module` and `python`) are replaced by
    one that only records its command line. They are created in `directory`, as the home
    directory of the script. Returns the process and the commands that were recorded.
    """
    programs = directory / "bin"
    home = directory / "home"
    programs.mkdir(parents=True, exist_ok=True)
    home.mkdir(exist_ok=True)
    for name in (".profile", ".bashrc"):
        (home / name).touch()
    recorded = directory / "commands.txt"
    recorded.write_text("")
    for name in ("srun", "conda", "module", "python"):
        program = programs / name
        program.write_text(f"#!/bin/sh\necho \"{name} $*\" >> '{recorded}'\n")
        program.chmod(0o755)

    # Nothing of the environment of the tests is given to the script
    environ = {"PATH": f"{programs}{os.pathsep}/usr/bin{os.pathsep}/bin", "HOME": str(home), "SLURM_ARRAY_TASK_ID": str(task_id)}
    completed = subprocess.run(["bash", str(script)], cwd=cwd, env=environ, capture_output=True, text=True, timeout=60)
    return completed, recorded.read_text().splitlines()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
