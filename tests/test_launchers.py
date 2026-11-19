"""Tests of the launchers (`recipes/<recipe>/scripts/run*.sh`): they can be run from any directory.

`python` is replaced by a program that records how it is called: nothing is trained.
"""

import os
import subprocess
import sys

import pytest

from helpers import ALL_TASKS, RECIPES, REPO, relative_path

MODEL = "some org/a model"  # with a space, to check that it is given as one argument
EXTRA_ARGUMENTS = ["--seed", "3", "--epochs", "2"]


def task_ids(task):
    return task.name


@pytest.fixture
def launch(tmp_path):
    """Run the launcher of a task from a directory.

    `launch(task, cwd, launcher)` returns the directory `python` is started in and its
    arguments. `launcher` is the launcher as it is typed from `cwd`, in one or several words.
    """
    programs = tmp_path / "bin"
    programs.mkdir()
    recorded = tmp_path / "recorded.txt"
    python = programs / "python"
    python.write_text(
        "#!/bin/sh\n"
        f"{{ pwd; for argument in \"$@\"; do printf '%s\\n' \"$argument\"; done; }} > '{recorded}'\n"
    )
    python.chmod(0o755)
    environ = dict(os.environ, PATH=f"{programs}{os.pathsep}{os.environ['PATH']}")

    def launch(task, cwd, launcher):
        arguments = [MODEL] + ([task.cli_subset] if task.cli_subset else []) + EXTRA_ARGUMENTS
        completed = subprocess.run(
            [*launcher, *arguments], cwd=cwd, env=environ, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        directory, *command_line = recorded.read_text().splitlines()
        return directory, command_line

    return launch


def ways_to_launch(task, elsewhere):
    """{description: (directory the launcher is run from, launcher as it is typed there)}"""
    other_recipe = "essai" if task.recipe != "essai" else "cas"
    other_scripts = RECIPES / other_recipe / "scripts"
    return {
        "from scripts/, as run.py does": (task.scripts_dir, [f"./{task.launcher}"]),
        "from scripts/, as the scripts for SLURM do": (task.scripts_dir, ["bash", task.launcher]),
        "from the root of the repository": (REPO, [relative_path(task.launcher_path, REPO)]),
        "from the root of the repository, with bash": (REPO, ["bash", relative_path(task.launcher_path, REPO)]),
        "from a directory out of the repository": (elsewhere, [str(task.launcher_path)]),
        # Its `../yaml/` directory has files with the same names as the one of the task
        "from the scripts of another recipe": (other_scripts, [relative_path(task.launcher_path, other_scripts)]),
    }


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_launcher_finds_its_files_from_any_directory(task, launch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    for description, (cwd, launcher) in ways_to_launch(task, elsewhere).items():
        directory, (script, config, model, *others) = launch(task, cwd, launcher)

        # The launcher does not change the current directory: the paths typed by the user stay valid
        assert os.path.samefile(directory, cwd), description
        assert os.path.samefile(os.path.join(cwd, script), task.scripts_dir / task.script), description
        assert config.startswith("--config="), description
        assert os.path.samefile(os.path.join(cwd, config[len("--config="):]), task.config_path), description
        assert model == f"--model={MODEL}", description
        subset = [f"--subset={task.cli_subset}"] if task.cli_subset else []
        assert others == subset + EXTRA_ARGUMENTS, description


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_run_has_the_same_name_from_any_directory(task, launch, tmp_path, utils, monkeypatch):
    """The command line of the launcher gives the same arguments, hence the same name to the run."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def parse(cwd, launcher):
        directory, command_line = launch(task, cwd, launcher)
        monkeypatch.chdir(directory)
        monkeypatch.setattr(sys, "argv", command_line)
        return utils.parse_args()

    (_, (cwd, launcher)), *others = ways_to_launch(task, elsewhere).items()
    reference = parse(cwd, launcher)
    assert reference.config == task.config_arg
    assert (reference.seed, reference.epochs) == (3, 2)

    for description, (cwd, launcher) in others:
        args = parse(cwd, launcher)
        assert args == reference, description
        assert utils.create_run_name(args) == utils.create_run_name(reference), description
