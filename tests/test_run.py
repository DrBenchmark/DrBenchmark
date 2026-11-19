"""Tests of `run.py`, with fake launchers instead of the real recipes."""

import os
import shutil
import subprocess
import sys

import pytest

from helpers import REPO, TASKS, Task

MODEL = "some org/a model"  # with a space, to check that it is given as one argument


def run_py(*arguments, cwd=REPO, script=REPO / "run.py"):
    return subprocess.run(
        [sys.executable, str(script), *arguments], cwd=cwd, capture_output=True, text=True, timeout=120,
    )


class FakeRepository:
    """A directory with a copy of `run.py` and launchers that only record how they are called.

    `run.py` finds the recipes and `models.txt` from its own location: its copy uses the fake ones.
    """

    def __init__(self, directory):
        self.directory = directory
        self.calls = directory / "calls.txt"
        directory.mkdir()
        shutil.copy(REPO / "run.py", directory)

    def scripts_dir(self, task_name):
        return self.directory / "recipes" / Task(task_name).recipe / "scripts"

    def create_launchers(self, exit_codes):
        """Create the launcher of each task of `exit_codes`, with the exit code given for it."""
        for task_name, exit_code in exit_codes.items():
            scripts_dir = self.scripts_dir(task_name)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            launcher = scripts_dir / Task(task_name).launcher
            launcher.write_text(
                "#!/bin/sh\n"
                f"printf '%s|%s|%s|%s|%s\\n' '{task_name}' \"$#\" \"$1\" \"$2\" \"$(pwd -P)\" >> '{self.calls}'\n"
                f"exit {exit_code}\n"
            )
            launcher.chmod(0o755)

    def run(self, *arguments, cwd=None):
        """Run the copy of `run.py`, from the root of the fake repository or from `cwd`."""
        return run_py(*arguments, cwd=self.directory if cwd is None else cwd, script=self.directory / "run.py")

    def recorded_calls(self):
        """[(task, number of arguments, model, subset, directory the launcher is run in)]"""
        return [tuple(line.split("|")) for line in self.calls.read_text().splitlines()]


@pytest.fixture
def fake_repository(tmp_path):
    return FakeRepository(tmp_path / "repository")


@pytest.fixture
def elsewhere(tmp_path):
    """A directory out of the repository, to run `run.py` from."""
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    return directory


def test_help_lists_every_task():
    usage = run_py("--help")
    assert usage.returncode == 0
    listed = usage.stdout.replace("\n", "").replace(" ", "")
    assert all(name in listed for name in TASKS)


def test_unknown_task_is_refused():
    unknown = run_py("--tasks", "unknown-task", "--models", MODEL)
    assert unknown.returncode == 2
    assert "invalid choice" in unknown.stderr


def test_runs_every_task_model_and_iteration(fake_repository):
    fake_repository.create_launchers({"cas-pos": 0, "e3c-ner_clinical": 0})
    completed = fake_repository.run(
        "--tasks", "cas-pos", "e3c-ner_clinical", "--models", MODEL, "other/model", "--nb-run", "2",
    )
    assert completed.returncode == 0, completed.stderr
    assert "failed" not in completed.stderr

    # task, number of arguments, model, subset
    assert [call[:4] for call in fake_repository.recorded_calls()] == [
        ("cas-pos", "1", MODEL, ""), ("cas-pos", "1", MODEL, ""),
        ("cas-pos", "1", "other/model", ""), ("cas-pos", "1", "other/model", ""),
        ("e3c-ner_clinical", "2", MODEL, "French_clinical"), ("e3c-ner_clinical", "2", MODEL, "French_clinical"),
        ("e3c-ner_clinical", "2", "other/model", "French_clinical"),
        ("e3c-ner_clinical", "2", "other/model", "French_clinical"),
    ]


def test_failed_runs_are_reported(fake_repository):
    """A failed run does not stop the next ones, but `run.py` lists it and exits with an error."""
    fake_repository.create_launchers({"cas-pos": 0, "cas-cls": 3, "clister-regr": 0})
    completed = fake_repository.run(
        "--tasks", "cas-pos", "cas-cls", "clister-regr", "--models", MODEL, "--nb-run", "2",
    )

    assert [call[0] for call in fake_repository.recorded_calls()] == [
        "cas-pos", "cas-pos", "cas-cls", "cas-cls", "clister-regr", "clister-regr",
    ]
    assert completed.returncode == 1
    assert "2 run(s) failed" in completed.stderr
    for iteration in (0, 1):
        assert f"cas-cls | {MODEL} | run {iteration} | exit code 3" in completed.stderr
    assert "cas-pos |" not in completed.stderr


def test_runs_from_any_directory(fake_repository, elsewhere):
    """Each launcher is run in its `scripts/` directory, wherever `run.py` is run from."""
    tasks = ["cas-pos", "e3c-ner_clinical"]
    fake_repository.create_launchers(dict.fromkeys(tasks, 0))
    expected = [
        ("cas-pos", "1", MODEL, "", os.path.realpath(fake_repository.scripts_dir("cas-pos"))),
        ("e3c-ner_clinical", "2", MODEL, "French_clinical", os.path.realpath(fake_repository.scripts_dir("e3c-ner_clinical"))),
    ]

    for cwd in (None, elsewhere, fake_repository.scripts_dir("cas-pos")):
        fake_repository.calls.write_text("")
        completed = fake_repository.run("--tasks", *tasks, "--models", MODEL, cwd=cwd)
        assert completed.returncode == 0, completed.stderr
        assert fake_repository.recorded_calls() == expected, cwd


def test_models_file_of_the_repository_is_read(fake_repository, elsewhere):
    """Without `--models`, the models are the ones of the `models.txt` which is next to `run.py`."""
    fake_repository.create_launchers({"cas-pos": 0})
    (fake_repository.directory / "models.txt").write_text("# a comment\nfirst/model\n\nsecond/model")
    (elsewhere / "models.txt").write_text("another/model\n")

    for cwd in (None, elsewhere):
        fake_repository.calls.write_text("")
        completed = fake_repository.run("--tasks", "cas-pos", cwd=cwd)
        assert completed.returncode == 0, completed.stderr
        assert [call[2] for call in fake_repository.recorded_calls()] == ["first/model", "second/model"], cwd


def test_local_model_typed_from_another_directory(fake_repository, elsewhere):
    """A relative path is relative to the current directory: the recipes get it from the root of the repository."""
    fake_repository.create_launchers({"cas-pos": 0})
    (elsewhere / "my_model").mkdir()

    completed = fake_repository.run(
        "--tasks", "cas-pos", "--models", "my_model", "my_model/", "a_hub/model", str(elsewhere / "my_model"),
        cwd=elsewhere,
    )
    assert completed.returncode == 0, completed.stderr
    assert [call[2] for call in fake_repository.recorded_calls()] == [
        # `utils.parse_args` looks for a relative path from the root of the repository
        os.path.join("..", "elsewhere", "my_model"),
        os.path.join("..", "elsewhere", "my_model") + "/",
        # A name of the HuggingFace Hub and an absolute path do not change
        "a_hub/model",
        str(elsewhere / "my_model"),
    ]


def test_local_model_typed_from_the_root_of_the_repository(fake_repository):
    """From the root of the repository, the models are given as they are typed, as before."""
    fake_repository.create_launchers({"cas-pos": 0})
    (fake_repository.directory / "my_model").mkdir()

    completed = fake_repository.run("--tasks", "cas-pos", "--models", "./my_model/", "my_model")
    assert completed.returncode == 0, completed.stderr
    assert [call[2] for call in fake_repository.recorded_calls()] == ["./my_model/", "my_model"]
