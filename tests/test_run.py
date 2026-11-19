"""Tests of `run.py`, with fake launchers instead of the real recipes."""

import subprocess
import sys

import pytest

from helpers import REPO, TASKS, Task

MODEL = "some org/a model"  # with a space, to check that it is given as one argument


def run_py(*arguments, cwd=REPO):
    return subprocess.run(
        [sys.executable, str(REPO / "run.py"), *arguments], cwd=cwd, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture
def fake_recipes(tmp_path):
    """A directory where `run.py` finds launchers that only record how they are called.

    `fake_recipes(exit_codes)` creates the launcher of each task of `exit_codes` with
    the exit code given for it, and returns the file where the calls are recorded.
    """
    calls = tmp_path / "calls.txt"

    def create(exit_codes):
        for task_name, exit_code in exit_codes.items():
            task = Task(task_name)
            scripts_dir = tmp_path / "recipes" / task.recipe / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            launcher = scripts_dir / task.launcher
            launcher.write_text(
                "#!/bin/sh\n"
                f"printf '%s|%s|%s|%s\\n' '{task_name}' \"$#\" \"$1\" \"$2\" >> '{calls}'\n"
                f"exit {exit_code}\n"
            )
            launcher.chmod(0o755)
        return calls

    return create


def test_help_lists_every_task():
    usage = run_py("--help")
    assert usage.returncode == 0
    listed = usage.stdout.replace("\n", "").replace(" ", "")
    assert all(name in listed for name in TASKS)


def test_unknown_task_is_refused():
    unknown = run_py("--tasks", "unknown-task", "--models", MODEL)
    assert unknown.returncode == 2
    assert "invalid choice" in unknown.stderr


def test_runs_every_task_model_and_iteration(fake_recipes, tmp_path):
    calls = fake_recipes({"cas-pos": 0, "e3c-ner_clinical": 0})
    completed = run_py("--tasks", "cas-pos", "e3c-ner_clinical", "--models", MODEL, "other/model", "--nb-run", "2",
                       cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "failed" not in completed.stderr

    # task | number of arguments | model | subset
    assert calls.read_text().splitlines() == [
        f"cas-pos|1|{MODEL}|", f"cas-pos|1|{MODEL}|",
        "cas-pos|1|other/model|", "cas-pos|1|other/model|",
        f"e3c-ner_clinical|2|{MODEL}|French_clinical", f"e3c-ner_clinical|2|{MODEL}|French_clinical",
        "e3c-ner_clinical|2|other/model|French_clinical", "e3c-ner_clinical|2|other/model|French_clinical",
    ]


def test_failed_runs_are_reported(fake_recipes, tmp_path):
    """A failed run does not stop the next ones, but `run.py` lists it and exits with an error."""
    calls = fake_recipes({"cas-pos": 0, "cas-cls": 3, "clister-regr": 0})
    completed = run_py("--tasks", "cas-pos", "cas-cls", "clister-regr", "--models", MODEL, "--nb-run", "2",
                       cwd=tmp_path)

    assert [line.split("|")[0] for line in calls.read_text().splitlines()] == [
        "cas-pos", "cas-pos", "cas-cls", "cas-cls", "clister-regr", "clister-regr",
    ]
    assert completed.returncode == 1
    assert "2 run(s) failed" in completed.stderr
    for iteration in (0, 1):
        assert f"cas-cls | {MODEL} | run {iteration} | exit code 3" in completed.stderr
    assert "cas-pos |" not in completed.stderr
