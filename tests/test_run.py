"""Tests of `run.py`, with fake launchers instead of the real recipes, but for one test."""

import argparse
import collections
import os
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from helpers import (
    MODELS_FILE, MODELS_OF_THE_FILE, RECIPES, REPO, TASKS, Task, load_module, read_json, run_command, run_without_slurm,
)

MODEL = "some org/a model"  # with a space, to check that it is given as one argument

# What a launcher was given: the model, the subset if the launcher takes one, then the options
Call = collections.namedtuple("Call", "task arguments directory")


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
        # `run.py` reads the options of the recipes in the code they share
        (directory / "recipes" / "common").mkdir(parents=True)
        shutil.copy(RECIPES / "common" / "utils.py", directory / "recipes" / "common")

    def scripts_dir(self, task_name):
        return self.directory / "recipes" / Task(task_name).recipe / "scripts"

    def create_launchers(self, exit_codes, already_computed=()):
        """Create the launcher of each task of `exit_codes`, with the exit code given for it.

        The launchers of the tasks of `already_computed` say, as a recipe does, that the
        results of their run exist.
        """
        for task_name, exit_code in exit_codes.items():
            scripts_dir = self.scripts_dir(task_name)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            launcher = scripts_dir / Task(task_name).launcher
            status = ""
            if task_name in already_computed:
                status = """printf '{"status": "already computed"}' > "$DRBENCHMARK_RUN_STATUS"\n"""
            launcher.write_text(
                "#!/bin/sh\n"
                # One line per call: the task, the current directory, then each argument, all followed by a tab
                f"{{ printf '%s\\t' '{task_name}' \"$(pwd -P)\" \"$@\"; echo; }} >> '{self.calls}'\n"
                f"{status}"
                f"exit {exit_code}\n"
            )
            launcher.chmod(0o755)

    def run(self, *arguments, cwd=None):
        """Run the copy of `run.py`, from the root of the fake repository or from `cwd`."""
        return run_py(*arguments, cwd=self.directory if cwd is None else cwd, script=self.directory / "run.py")

    def recorded_calls(self):
        """The calls of the launchers, in the order they were made."""
        calls = []
        for line in self.calls.read_text().splitlines():
            task, directory, *arguments = line.split("\t")[:-1]
            calls.append(Call(task, arguments, directory))
        return calls

    def models(self):
        """The model of each call."""
        return [call.arguments[0] for call in self.recorded_calls()]


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
    """Each run of a task for a model has its seed: 0 for the first one, then 1..."""
    fake_repository.create_launchers({"cas-pos": 0, "e3c-ner_clinical": 0})
    completed = fake_repository.run(
        "--tasks", "cas-pos", "e3c-ner_clinical", "--models", MODEL, "other/model", "--nb-run", "2",
    )
    assert completed.returncode == 0, completed.stderr
    assert "ERROR" not in completed.stderr
    assert "INFO::8 run(s): 8 computed, 0 already computed, 0 failed" in completed.stderr

    assert [(call.task, call.arguments) for call in fake_repository.recorded_calls()] == [
        ("cas-pos", [MODEL, "--seed", "0"]), ("cas-pos", [MODEL, "--seed", "1"]),
        ("cas-pos", ["other/model", "--seed", "0"]), ("cas-pos", ["other/model", "--seed", "1"]),
        ("e3c-ner_clinical", [MODEL, "French_clinical", "--seed", "0"]),
        ("e3c-ner_clinical", [MODEL, "French_clinical", "--seed", "1"]),
        ("e3c-ner_clinical", ["other/model", "French_clinical", "--seed", "0"]),
        ("e3c-ner_clinical", ["other/model", "French_clinical", "--seed", "1"]),
    ]


def test_runs_are_the_same_ones_each_time(fake_repository):
    """`run.py` asks for the same runs when it is started again: the recipes skip the ones they already computed."""
    fake_repository.create_launchers({"cas-pos": 0})
    for _ in range(2):
        assert fake_repository.run("--tasks", "cas-pos", "--models", MODEL, "--nb-run", "2").returncode == 0
    twice = [call.arguments for call in fake_repository.recorded_calls()]
    assert twice[:2] == twice[2:] == [[MODEL, "--seed", "0"], [MODEL, "--seed", "1"]]


def test_seed_of_the_first_run_can_be_chosen(fake_repository):
    fake_repository.create_launchers({"cas-pos": 0})
    completed = fake_repository.run("--tasks", "cas-pos", "--models", MODEL, "--nb-run", "3", "--seed", "40")
    assert completed.returncode == 0, completed.stderr
    assert [call.arguments for call in fake_repository.recorded_calls()] == [
        [MODEL, "--seed", "40"], [MODEL, "--seed", "41"], [MODEL, "--seed", "42"],
    ]
    for iteration, seed in enumerate((40, 41, 42)):
        assert f"(run {iteration}, seed {seed})" in completed.stderr


def test_job_is_one_of_the_runs(fake_repository):
    """With `--job`, `run.py` only does the run that has this number, among the ones it does without it."""
    fake_repository.create_launchers({"cas-pos": 0, "e3c-ner_clinical": 0})
    asked = ("--tasks", "cas-pos", "e3c-ner_clinical", "--models", MODEL, "other/model", "--nb-run", "2", "--epochs", "3")

    assert fake_repository.run(*asked).returncode == 0
    runs = fake_repository.recorded_calls()
    assert len(runs) == 8

    for job, run in enumerate(runs):
        fake_repository.calls.write_text("")
        completed = fake_repository.run(*asked, "--job", str(job))
        assert completed.returncode == 0, completed.stderr
        assert fake_repository.recorded_calls() == [run], job
        assert "There is no run" not in completed.stderr

    # The job of a job array that is larger than the number of runs has nothing to do
    fake_repository.calls.write_text("")
    completed = fake_repository.run(*asked, "--job", "8")
    assert completed.returncode == 0
    assert "There is no run 8: 8 runs are asked for, numbered from 0" in completed.stderr
    assert fake_repository.recorded_calls() == []

    refused = fake_repository.run(*asked, "--job", "-1")
    assert refused.returncode == 2
    assert "the runs are numbered from 0: --job -1" in refused.stderr


def test_failure_of_a_job_is_the_one_of_its_run(fake_repository):
    fake_repository.create_launchers({"cas-pos": 0, "cas-cls": 3})
    asked = ("--tasks", "cas-pos", "cas-cls", "--models", MODEL)
    assert fake_repository.run(*asked, "--job", "0").returncode == 0
    assert fake_repository.run(*asked, "--job", "1").returncode == 1


@pytest.mark.parametrize("script", ["run_all_slurm.sh", "run_all_jean_zay.sh"])
def test_job_of_a_script_for_slurm_does_its_run(script, fake_repository, tmp_path):
    """The scripts for SLURM run `run.py` for real: each job does one run of a task for a model of `models.txt`."""
    fake_repository.create_launchers(dict.fromkeys(TASKS, 0))
    (fake_repository.directory / "models.txt").write_text(MODELS_FILE)
    nb_runs = int(re.search(r"^NBR_RUNS=(\d+)$", (REPO / script).read_text(), re.M).group(1))
    # Task by task, then model by model: `run.py` does the tasks in the order of their names
    runs = [(task, model, run) for task in sorted(TASKS) for model in MODELS_OF_THE_FILE for run in range(nb_runs)]

    def done(job, last_job=len(runs) - 1):
        fake_repository.calls.write_text("")
        completed, _ = run_without_slurm(
            REPO / script, cwd=fake_repository.directory, directory=tmp_path / "slurm",
            task_id=job, last_task_id=last_job, run_python=True,
        )
        assert completed.returncode == 0, completed.stderr
        calls = [(call.task, call.arguments[0], int(call.arguments[-1])) for call in fake_repository.recorded_calls()]
        return calls, completed.stderr

    for job in (0, 1, nb_runs, 2 * nb_runs, len(runs) - 1):
        calls, messages = done(job)
        # The seed of the run is its number for the task and the model
        assert calls == [runs[job]], job
        assert "job array" not in messages
    calls, messages = done(len(runs))
    assert calls == []
    assert f"There is no run {len(runs)}: {len(runs)} runs are asked for, numbered from 0" in messages

    # A job array that is too small does not do the last runs: each of its jobs says so
    calls, messages = done(0, last_job=len(runs) - 3)
    assert calls == [runs[0]]
    assert (
        f"The job array stops at the job {len(runs) - 3}: the runs {len(runs) - 2} to {len(runs) - 1} have no job. "
        f"There is one job per run with --array=0-{len(runs) - 1}"
    ) in messages


def test_options_of_the_recipes_are_given_to_every_run(fake_repository, utils):
    fake_repository.create_launchers({"cas-pos": 0, "e3c-ner_clinical": 0})
    completed = fake_repository.run(
        "--tasks", "cas-pos", "e3c-ner_clinical", "--models", MODEL, "--nb-run", "2",
        "--offline", "false", "--max_train_samples", "10", "--epochs", "2", "--fewshot", "0.25", "--learning_rate", "2e-5",
    )
    assert completed.returncode == 0, completed.stderr

    # The seed and the five options, with their values: what the launchers are given after
    # the model and the subset, and give to the script of the recipe
    calls = fake_repository.recorded_calls()
    assert [(call.task, call.arguments[:-12]) for call in calls] == [
        ("cas-pos", [MODEL]), ("cas-pos", [MODEL]),
        ("e3c-ner_clinical", [MODEL, "French_clinical"]), ("e3c-ner_clinical", [MODEL, "French_clinical"]),
    ]
    # The recipes read in them the values that were typed
    parser = argparse.ArgumentParser()
    names = utils.add_arguments(parser, but=("config",))
    for call, seed in zip(calls, (0, 1, 0, 1)):
        read = vars(parser.parse_args(call.arguments[-12:]))
        typed = {"seed": seed, "offline": False, "max_train_samples": 10, "epochs": 2, "fewshot": 0.25, "learning_rate": 2e-5}
        assert read == {name: typed.get(name) for name in names}


@pytest.mark.parametrize("options, message", [
    (["--epochz", "1"], "unrecognized arguments: --epochz 1"),
    (["--epochs", "many"], "argument --epochs: invalid int value: 'many'"),
    (["--offline", "maybe"], "argument --offline: expected true or false, got 'maybe'"),
    # `run.py` gives the subset of each task itself, and each recipe has its own directories
    (["--subset", "emea"], "unrecognized arguments: --subset emea"),
    (["--run_dir", "somewhere"], "unrecognized arguments: --run_dir somewhere"),
    (["--seed", "-1"], "the seeds of the runs have to be between 0 and 2**32-1"),
    (["--seed", str(2**32 - 1), "--nb-run", "2"], "the seeds of the runs have to be between 0 and 2**32-1"),
], ids=lambda value: " ".join(value) if isinstance(value, list) else None)
def test_wrong_option_is_refused_before_any_run(options, message, fake_repository):
    """An option that the recipes would refuse is not given to each run, to fail each time."""
    fake_repository.create_launchers({"cas-pos": 0})
    completed = fake_repository.run("--tasks", "cas-pos", "--models", MODEL, *options)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert not fake_repository.calls.exists()


def test_run_py_has_the_options_of_the_recipes(utils):
    """Every option of the recipes is one of `run.py`, but the ones it sets itself and the directories."""
    not_options = load_module("drbenchmark_run", REPO / "run.py").NOT_OPTIONS_OF_RUN_PY
    options_of_the_recipes = utils.add_arguments(argparse.ArgumentParser())
    assert set(not_options) < set(options_of_the_recipes)

    usage = run_py("--help")
    assert usage.returncode == 0
    listed = re.findall(r"^  --(\w+)", usage.stdout.split("options of the recipes:")[1], re.M)
    assert listed == [name for name in options_of_the_recipes if name not in not_options]
    assert {"epochs", "offline", "fewshot"} <= set(listed)


def test_run_py_loads_neither_torch_nor_transformers():
    """`run.py` reads the options of the recipes in `recipes/common/utils.py`, which has to stay quick to load."""
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(REPO / "run.py"), "--help"], capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    # import time:       105 |        105 |   yaml.error
    imported = {name.split(".")[0] for name in re.findall(r"^import time:\s+\d+ \|\s+\d+ \|\s+(\S+)$", completed.stderr, re.M)}
    assert "yaml" in imported
    assert not imported & {"torch", "transformers", "datasets"}


def test_failed_runs_are_reported(fake_repository):
    """A failed run does not stop the next ones, but `run.py` lists it and exits with an error."""
    fake_repository.create_launchers({"cas-pos": 0, "cas-cls": 3, "clister-regr": 0})
    completed = fake_repository.run(
        "--tasks", "cas-pos", "cas-cls", "clister-regr", "--models", MODEL, "--nb-run", "2",
    )

    assert [call.task for call in fake_repository.recorded_calls()] == [
        "cas-pos", "cas-pos", "cas-cls", "cas-cls", "clister-regr", "clister-regr",
    ]
    assert completed.returncode == 1
    assert "2 run(s) failed" in completed.stderr
    for iteration in (0, 1):
        assert f"cas-cls | {MODEL} | run {iteration} | exit code 3" in completed.stderr
    assert "cas-pos |" not in completed.stderr


def test_runs_are_counted(fake_repository):
    """At its end, `run.py` says how many runs were computed, were already computed and failed."""
    fake_repository.create_launchers({"cas-pos": 0, "cas-cls": 0, "deft2020-cls": 0, "clister-regr": 3}, already_computed=["cas-cls"])

    # The runs that follow the ones that were already computed are not taken for such runs
    completed = fake_repository.run(
        "--tasks", "cas-pos", "cas-cls", "deft2020-cls", "clister-regr", "--models", MODEL, "--nb-run", "2",
    )
    assert completed.returncode == 1
    assert "INFO::8 run(s): 4 computed, 2 already computed, 2 failed" in completed.stderr

    # A run that ends well without saying what it did was computed
    completed = fake_repository.run("--tasks", "cas-pos", "--models", MODEL)
    assert completed.returncode == 0
    assert "INFO::1 run(s): 1 computed, 0 already computed, 0 failed" in completed.stderr
    completed = fake_repository.run("--tasks", "cas-cls", "--models", MODEL, "--job", "0")
    assert "INFO::1 run(s): 0 computed, 1 already computed, 0 failed" in completed.stderr


def test_run_says_what_it_did_to_run_py(utils, monkeypatch, tmp_path):
    """A recipe writes what its run did where `run.py` asks for it, and nowhere when nothing asks."""
    monkeypatch.delenv(utils.STATUS_FILE, raising=False)
    monkeypatch.chdir(tmp_path)
    utils.report_status(utils.COMPUTED, "runs/some_run.json")
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv(utils.STATUS_FILE, str(tmp_path / "status.json"))
    utils.report_status(utils.ALREADY_COMPUTED, "runs/some_run.json")
    assert read_json(tmp_path / "status.json") == {"status": "already computed", "results": str(tmp_path / "runs" / "some_run.json")}


def test_models_file_is_only_needed_without_models(fake_repository):
    """`run.py` does not warn about a missing `models.txt` when the models are given, and shows its messages in both cases."""
    fake_repository.create_launchers({"cas-pos": 0})
    assert not (fake_repository.directory / "models.txt").exists()

    completed = fake_repository.run("--tasks", "cas-pos", "--models", MODEL)
    assert completed.returncode == 0
    assert "models.txt" not in completed.stderr
    assert f"INFO::Evaluating {MODEL} on cas-pos (run 0, seed 0)" in completed.stderr

    completed = fake_repository.run("--tasks", "cas-pos")
    assert completed.returncode == 0
    assert "WARNING::`models.txt` not found. Please provide models in --models" in completed.stderr
    assert "INFO::0 run(s): 0 computed, 0 already computed, 0 failed" in completed.stderr
    assert len(fake_repository.recorded_calls()) == 1


def test_runs_from_any_directory(fake_repository, elsewhere):
    """Each launcher is run in its `scripts/` directory, wherever `run.py` is run from."""
    tasks = ["cas-pos", "e3c-ner_clinical"]
    fake_repository.create_launchers(dict.fromkeys(tasks, 0))
    expected = [
        Call("cas-pos", [MODEL, "--seed", "0"], os.path.realpath(fake_repository.scripts_dir("cas-pos"))),
        Call(
            "e3c-ner_clinical", [MODEL, "French_clinical", "--seed", "0"],
            os.path.realpath(fake_repository.scripts_dir("e3c-ner_clinical")),
        ),
    ]

    for cwd in (None, elsewhere, fake_repository.scripts_dir("cas-pos")):
        fake_repository.calls.write_text("")
        completed = fake_repository.run("--tasks", *tasks, "--models", MODEL, cwd=cwd)
        assert completed.returncode == 0, completed.stderr
        assert fake_repository.recorded_calls() == expected, cwd


def test_models_file_of_the_repository_is_read(fake_repository, elsewhere):
    """Without `--models`, the models are the ones of the `models.txt` which is next to `run.py`.

    Neither its empty lines nor its comments are models.
    """
    fake_repository.create_launchers({"cas-pos": 0})
    (fake_repository.directory / "models.txt").write_text(MODELS_FILE)
    (elsewhere / "models.txt").write_text("another/model\n")

    for cwd in (None, elsewhere):
        fake_repository.calls.write_text("")
        completed = fake_repository.run("--tasks", "cas-pos", cwd=cwd)
        assert completed.returncode == 0, completed.stderr
        assert fake_repository.models() == MODELS_OF_THE_FILE, cwd


def test_local_model_typed_from_another_directory(fake_repository, elsewhere):
    """A relative path is relative to the current directory: the recipes get it from the root of the repository."""
    fake_repository.create_launchers({"cas-pos": 0})
    (elsewhere / "my_model").mkdir()

    completed = fake_repository.run(
        "--tasks", "cas-pos", "--models", "my_model", "my_model/", "a_hub/model", str(elsewhere / "my_model"),
        cwd=elsewhere,
    )
    assert completed.returncode == 0, completed.stderr
    assert fake_repository.models() == [
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
    assert fake_repository.models() == ["./my_model/", "my_model"]


@pytest.fixture
def small_repository(tmp_path, synthetic_data):
    """What `run.py` needs to run the tasks of CAS for real, on the synthetic data.

    The copy of `run.py` runs the copy of the recipe, which reads its data and writes its
    results in the directories of its YAML file, as in the repository.
    """
    repository = tmp_path / "repository"
    recipe = repository / "recipes" / "cas"
    recipe.mkdir(parents=True)
    for name in ("run.py", "config.yaml"):
        shutil.copy(REPO / name, repository)
    not_copied = shutil.ignore_patterns("__pycache__", "logs")
    # The recipes load their metrics from the root of the repository
    shutil.copytree(REPO / "metrics", repository / "metrics", ignore=not_copied)
    shutil.copytree(RECIPES / "common", repository / "recipes" / "common", ignore=not_copied)
    for name in ("scripts", "yaml"):
        shutil.copytree(RECIPES / "cas" / name, recipe / name, ignore=not_copied)
    # Where `download_datasets_locally.py` saves the datasets: `data/local_hf_<subset>/`
    os.symlink(synthetic_data / "cas", recipe / "data")
    (recipe / "runs").mkdir()
    return repository


@pytest.mark.e2e
def test_run_already_computed_is_not_computed_again(small_repository, tiny_bert, elsewhere):
    """Started again, `run.py` only computes the runs that are missing: here, the third one."""
    def run(*arguments):
        command = [sys.executable, small_repository / "run.py", "--tasks", "cas-pos", "--models", tiny_bert, *arguments]
        completed = run_command(" ".join(shlex.quote(str(argument)) for argument in command), cwd=elsewhere)
        assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
        results = {}
        for path in (small_repository / "recipes" / "cas" / "runs").glob("*.json"):
            results[read_json(path)["hyperparameters"]["seed"]] = path
        return completed, results

    options = ("--epochs", "1", "--offline", "true", "--max_train_samples", "8")
    completed, first = run("--nb-run", "2", *options)
    assert sorted(first) == [0, 1]
    assert "already computed" not in completed.stdout
    assert "INFO::2 run(s): 2 computed, 0 already computed, 0 failed" in completed.stderr
    for path in first.values():
        hyperparameters = read_json(path)["hyperparameters"]
        # The options of `run.py` were given to the recipe
        assert (hyperparameters["epochs"], hyperparameters["offline"], hyperparameters["max_train_samples"]) == (1, True, 8)
    written = {seed: path.stat().st_mtime_ns for seed, path in first.items()}

    completed, second = run("--nb-run", "3", *options)
    assert sorted(second) == [0, 1, 2]
    assert completed.stdout.count("This configuration was already computed") == 2
    assert "INFO::3 run(s): 1 computed, 2 already computed, 0 failed" in completed.stderr
    # The two first runs were left as they were
    assert {seed: second[seed] for seed in (0, 1)} == first
    assert {seed: second[seed].stat().st_mtime_ns for seed in (0, 1)} == written

    # With another value of an option, these are other runs
    completed, _ = run("--nb-run", "1", *options[:-1], "6")
    assert len(list((small_repository / "recipes" / "cas" / "runs").glob("*.json"))) == 4
    assert "already computed" not in completed.stdout
