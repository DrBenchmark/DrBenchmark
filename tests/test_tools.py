"""Tests of the tools that work on the results of the runs (`tools/`)."""

import ast
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from helpers import ALL_TASKS, REPO, copy_tools, load_module, read_json, run_tool, run_without_slurm

TOOLS = sorted(path.name for path in (REPO / "tools").glob("*.py"))

# Metrics collected by `tools/CollectAndAverageScores.py` for each kind of task
COLLECTED_METRICS = {
    "pos": {"overall_f1", "overall_accuracy"},
    "ner": {"overall_f1", "overall_accuracy"},
    "cls": {"macro_f1", "weighted_f1"},
    "mcqa": {"hamming_score", "exact_match"},
    "regr": {"edrm", "spearman_correlation_coef"},
}


def results_key(task):
    """Key of a task in `stats/results.json`, for the runs on the whole training set."""
    return f"{task.recipe}|{task.name.split('-')[1]}|1.0"


# Functions whose first argument is the path of a file or of a directory
PATH_FUNCTIONS = {"open", "makedirs", "listdir", "glob", "savefig", "to_excel", "to_csv", "read_csv", "read_excel"}


def beginning_of_literal(node):
    """What a string written in the code starts with: a constant, or the first part of an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return node.values[0].value
    return None


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_does_not_use_a_path_relative_to_the_current_directory(tool):
    """The tools find the files of the repository with `tools/repository_paths.py`: they can be run from any directory."""
    relative_paths = []
    for node in ast.walk(ast.parse((REPO / "tools" / tool).read_text())):
        if isinstance(node, ast.Call) and node.args:
            function = getattr(node.func, "id", getattr(node.func, "attr", None))
            path = beginning_of_literal(node.args[0])
            if function in PATH_FUNCTIONS and path is not None and not os.path.isabs(path):
                relative_paths.append(f"line {node.lineno}: {function}({path!r}")
    assert not relative_paths


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_does_nothing_when_it_is_imported(tool):
    """The work of a tool is in functions, or under `if __name__ == '__main__'`: it has options and can be tested."""
    tree = ast.parse((REPO / "tools" / tool).read_text())
    work = [f"line {node.lineno}" for node in tree.body if isinstance(node, (ast.For, ast.While, ast.With))]
    assert not work


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_is_documented(tool):
    """`tools/README.md` says what each tool does."""
    assert f"[{tool}](./{tool})" in (REPO / "tools" / "README.md").read_text()


@pytest.fixture(scope="module")
def results_repository(run_recipe, tiny_bert, tmp_path_factory):
    """A copy of the layout of the repository, with the tools and one run of every task."""
    repository = tmp_path_factory.mktemp("results")
    copy_tools(repository)
    for task in ALL_TASKS:
        run = run_recipe(task.name, tiny_bert)
        run.check()
        runs_directory = repository / "recipes" / task.recipe / "runs"
        runs_directory.mkdir(parents=True, exist_ok=True)
        for run_file in run.run_files:
            shutil.copy(run_file, runs_directory)
    return repository


@pytest.mark.e2e
def test_collect_and_average_scores(results_repository, tiny_bert, tmp_path):
    # The tool is run from another directory: it must find the results in its repository
    completed = run_tool("CollectAndAverageScores.py", results_repository, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert not list(tmp_path.iterdir())

    results = read_json(results_repository / "stats" / "results.json")
    averages = read_json(results_repository / "stats" / "overall_averaged_metrics.json")
    assert list(results) == list(averages) == [str(tiny_bert)]

    expected = {results_key(task): COLLECTED_METRICS[task.kind] for task in ALL_TASKS}
    for scores in (results[str(tiny_bert)], averages[str(tiny_bert)]):
        assert {key: set(metrics) for key, metrics in scores.items()} == expected
    # One run per task
    assert all(len(values) == 1 for metrics in results[str(tiny_bert)].values() for values in metrics.values())


@pytest.mark.e2e
def test_build_excel_table_of_scores(results_repository, tmp_path):
    assert run_tool("CollectAndAverageScores.py", results_repository, cwd=tmp_path).returncode == 0
    completed = run_tool("BuildExcelTableOfScores.py", results_repository, "--nb-run", "1", "--csv", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]

    lines = completed.stdout.strip().splitlines()
    assert lines[0].startswith("dataset,task,metric")
    # One line per score shown on the benchmark: every task but DEFT-2019,
    # and two scores for the question answering and similarity tasks
    two_scores = sum(task.kind in ("mcqa", "regr") for task in ALL_TASKS)
    assert len(lines) - 1 == len(ALL_TASKS) - 1 + two_scores

    # A path given on the command line is relative to the current directory, as usual
    shutil.copy(results_repository / "stats" / "results.json", tmp_path / "copy.json")
    with_path = run_tool(
        "BuildExcelTableOfScores.py", results_repository, "--nb-run", "1", "--csv", "--results", "copy.json", cwd=tmp_path,
    )
    assert with_path.returncode == 0, with_path.stderr[-3000:]
    assert with_path.stdout == completed.stdout


class RunsRepository:
    """A copy of the tools in a directory where runs are written by hand, to run `CollectAndAverageScores.py` on them."""

    def __init__(self, directory):
        self.repository = directory / "repository"
        self.repository.mkdir()
        copy_tools(self.repository)
        self.elsewhere = directory / "elsewhere"
        self.elsewhere.mkdir()
        self.stats = self.repository / "stats"

    def write_run(self, name, *, task="morfitt-cls", updates_per_epoch=None, scores=(0.5, 0.25), **hyperparameters):
        """Write the results of a run of a classification task, as its training script does.

        The hyper-parameters are the ones the first scripts wrote: `hyperparameters`
        replaces them or adds the ones of the later scripts. The state of the trainer,
        which the first scripts did not write either, is given by `updates_per_epoch`.
        """
        run = {
            "hyperparameters": {
                "task": task, "model_name": "../../../models/some_model", "fewshot": 1.0, "epochs": 35,
                "batch_size": 12, "learning_rate": 2e-05, "weight_decay": 0.01, "max_position_embeddings": 512,
                **hyperparameters,
            },
            "metrics": {"macro avg": {"f1-score": scores[0]}, "weighted avg": {"f1-score": scores[1]}},
        }
        if updates_per_epoch is not None:
            epochs = run["hyperparameters"]["epochs"]
            run["trainer_state"] = {"epoch": float(epochs), "global_step": updates_per_epoch * epochs}
        runs_directory = self.repository / "recipes" / task.split("-")[0] / "runs"
        runs_directory.mkdir(parents=True, exist_ok=True)
        (runs_directory / f"{name}.json").write_text(json.dumps(run))

    def collect_and_average_scores(self, *arguments):
        return run_tool("CollectAndAverageScores.py", self.repository, *arguments, cwd=self.elsewhere)

    def results(self):
        return read_json(self.stats / "results.json")


@pytest.fixture
def runs(tmp_path):
    return RunsRepository(tmp_path)


def test_runs_trained_in_the_same_way_are_averaged(runs):
    """The seed and the location of the files are what distinguishes the runs of a task."""
    runs.write_run("first", scores=(0.5, 0.25), seed=1, run_dir="../runs/", offline=True)
    runs.write_run("second", scores=(1.0, 0.75), seed=2, run_dir="/somewhere/else/", offline=False)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert {metric: sorted(values) for metric, values in runs.results()["some/model"]["morfitt|cls|1.0"].items()} == {
        "macro_f1": [0.5, 1.0], "weighted_f1": [0.25, 0.75],
    }
    averages = read_json(runs.stats / "overall_averaged_metrics.json")
    assert averages["some/model"]["morfitt|cls|1.0"] == {"macro_f1": 0.75, "weighted_f1": 0.5}


def test_runs_with_different_hyperparameters_are_not_averaged(runs):
    runs.write_run("ten_epochs", epochs=10)
    runs.write_run("ten_epochs_too", epochs=10)
    runs.write_run("three_epochs", epochs=3)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert not (runs.stats / "results.json").exists()
    assert not (runs.stats / "overall_averaged_metrics.json").exists()
    # The report says which runs differ, on what, and how to go on
    for expected in (
        "some/model | morfitt-cls | 1.0 of the training set",
        "    epochs\n        10: 2 run(s)\n",
        "        3: 1 run(s)\n            recipes/morfitt/runs/three_epochs.json\n",
        "1 set(s) of runs cannot be averaged",
        "--ignore epochs",
    ):
        assert expected in completed.stderr, completed.stderr

    # The difference can be accepted
    accepted = runs.collect_and_average_scores("--ignore", "epochs")
    assert accepted.returncode == 0, accepted.stderr[-3000:]
    assert len(runs.results()["some/model"]["morfitt|cls|1.0"]["macro_f1"]) == 3


def test_hyperparameter_which_is_not_set_is_a_value(runs):
    runs.write_run("truncated", max_position_embeddings=512)
    runs.write_run("not_truncated", max_position_embeddings=None)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert "    max_position_embeddings\n" in completed.stderr
    assert "        not set: 1 run(s)\n            recipes/morfitt/runs/not_truncated.json\n" in completed.stderr


def test_effective_batch_size_is_compared(runs):
    """12 examples per batch and 12 accumulated batches of one example are the same training."""
    runs.write_run("batches_of_12", batch_size=12)
    runs.write_run("accumulated", batch_size=1, gradient_accumulation_steps=12)
    runs.write_run("accumulated_by_4", batch_size=3, gradient_accumulation_steps=4)
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]

    # What the scripts did when they read the accumulation without applying it
    runs.write_run("one_example_per_update", batch_size=1, gradient_accumulation_steps=1)
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert "    effective_batch_size (batch_size x gradient_accumulation_steps)\n        12: 3 run(s)\n" in completed.stderr
    assert "        1: 1 run(s)\n            recipes/morfitt/runs/one_example_per_update.json\n" in completed.stderr
    assert "--ignore effective_batch_size" in completed.stderr


def test_updates_of_the_weights_are_compared(runs):
    """The state of the trainer says what the training did, even when the hyper-parameters are the same."""
    accumulation = {"batch_size": 1, "gradient_accumulation_steps": 12}
    runs.write_run("accumulated", updates_per_epoch=127, **accumulation)
    # One update more or less per epoch: the last batches of an epoch are not counted in the same way
    # by every version of transformers
    runs.write_run("accumulated_too", updates_per_epoch=126, **accumulation)
    # The first scripts did not write the state of the trainer: unknown is not different
    runs.write_run("first_scripts", batch_size=12)
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]

    # The accumulation is written in the hyper-parameters, but the weights were updated after each example
    runs.write_run("not_accumulated", updates_per_epoch=1514, **accumulation)
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert "    updates_per_epoch (measured: updates of the weights during the training / epochs)\n" in completed.stderr
    assert "        1514: 1 run(s)\n            recipes/morfitt/runs/not_accumulated.json\n" in completed.stderr
    assert "effective_batch_size" not in completed.stderr
    assert runs.collect_and_average_scores("--ignore", "updates_per_epoch").returncode == 0


def test_runs_of_different_versions_of_the_scripts_are_compared(runs):
    """The first scripts wrote the learning rate as a string, and had fewer arguments."""
    runs.write_run("first_scripts", learning_rate="2e-5")
    runs.write_run(
        "last_scripts", learning_rate=2e-05, seed=3, run_dir="../runs/", gradient_accumulation_steps=1,
        max_train_samples=None, max_val_samples=None, max_test_samples=None, updates_per_epoch=127,
    )
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]

    runs.write_run("another_learning_rate", learning_rate="1e-5")
    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert "    learning_rate\n        2e-05: 2 run(s)\n" in completed.stderr


def test_runs_of_a_model_are_gathered_under_its_name_on_the_hub(runs):
    """Online or offline, old or recent: the runs of a model are the runs of one model."""
    local_copy, name = "../../../models/dr-bert_drbert-7gb", "Dr-BERT/DrBERT-7GB"
    runs.write_run("recent_offline", scores=(0.1, 0.1), model_name=local_copy, hub_model_name=name)
    runs.write_run("recent_online", scores=(0.2, 0.2), model_name=name, hub_model_name=name)
    # The runs older than `hub_model_name`: the name is the one of the recent runs
    runs.write_run("old_offline", scores=(0.3, 0.3), model_name=local_copy)
    runs.write_run("old_online", scores=(0.4, 0.4), model_name=name)
    # The same model, named with other upper case letters
    runs.write_run("other_case", scores=(0.5, 0.5), model_name="dr-bert/DrBERT-7GB", hub_model_name="dr-bert/DrBERT-7GB")

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]
    results = runs.results()
    assert list(results) == [name]
    assert sorted(results[name]["morfitt|cls|1.0"]["macro_f1"]) == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert list(read_json(runs.stats / "overall_averaged_metrics.json")) == [name]


def test_name_of_an_old_run_is_looked_for_in_the_models_file(runs):
    """An old offline run only has the directory of its model, in lower case."""
    (runs.repository / "models.txt").write_text(
        "# Models to evaluate\nFacebookAI/xlm-roberta-base\n\n   # Not/This-One\nflaubert/flaubert_base_uncased"
    )
    runs.write_run("in_the_file", model_name="../../../models/facebookai_xlm-roberta-base")
    runs.write_run("underscores_in_the_name", model_name="../../../models/flaubert_flaubert_base_uncased")
    # Unknown: the name is rebuilt from the directory, in lower case
    runs.write_run("not_in_the_file", model_name="../../../models/dr-bert_drbert-7gb")
    # A model given by a path is named by its path
    runs.write_run("path", model_name="/somewhere/my_model", hub_model_name=None)
    runs.write_run("other_relative_path", model_name="../../../assets/my_model", hub_model_name=None)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert set(runs.results()) == {
        "FacebookAI/xlm-roberta-base", "flaubert/flaubert_base_uncased", "dr-bert/drbert-7gb",
        "/somewhere/my_model", "../../../assets/my_model",
    }


def test_runs_gathered_under_the_name_of_a_model_are_compared(runs):
    """Online and offline runs are averaged together: they have to be trained in the same way."""
    name = "Dr-BERT/DrBERT-7GB"
    runs.write_run("offline", epochs=10, model_name="../../../models/dr-bert_drbert-7gb", hub_model_name=name)
    runs.write_run("online", epochs=3, model_name=name, hub_model_name=name)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 1
    assert "Dr-BERT/DrBERT-7GB | morfitt-cls | 1.0 of the training set\n    epochs\n" in completed.stderr
    # The name of the model is not a difference between the runs
    assert "model_name" not in completed.stderr


def test_only_the_runs_that_are_averaged_together_are_compared(runs):
    runs.write_run("reference", epochs=10)
    runs.write_run("another_model", epochs=3, model_name="../../../models/another_model")
    runs.write_run("another_task", epochs=3, task="diamed-cls")
    runs.write_run("part_of_the_training_set", epochs=3, fewshot=0.5)

    completed = runs.collect_and_average_scores()
    assert completed.returncode == 0, completed.stderr[-3000:]
    results = runs.results()
    assert set(results) == {"some/model", "another/model"}
    assert set(results["some/model"]) == {"morfitt|cls|1.0", "diamed|cls|1.0", "morfitt|cls|0.5"}


@pytest.fixture
def write_slurm_script(tmp_path):
    """Run `tools/WriteSlurmScriptForMissingRuns.py` on the number of runs given for each task."""
    pytest.importorskip("liquid", reason="python-liquid is needed by tools/WriteSlurmScriptForMissingRuns.py")
    repository = tmp_path / "repository"
    repository.mkdir()
    copy_tools(repository)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def run(runs_per_task, *arguments):
        results = {
            results_key(task): {"overall_f1": [0.5] * runs_per_task[task.name]}
            for task in ALL_TASKS if runs_per_task.get(task.name)
        }
        (repository / "stats" / "results.json").write_text(json.dumps({"Dr-BERT/DrBERT-7GB": results}))
        return run_tool("WriteSlurmScriptForMissingRuns.py", repository, *arguments, cwd=elsewhere)

    return run


def test_write_slurm_script_for_missing_runs(write_slurm_script):
    """The generated job array has one index per missing run, for 4 runs per task."""
    completed = write_slurm_script({"cas-pos": 4, "cas-cls": 1})
    assert completed.returncode == 0, completed.stderr[-3000:]
    script = completed.stdout

    jobs = re.findall(r'^    "(cd recipes/\w+/scripts/ ; \./run\S*\.sh .+)"$', script, re.M)
    # Nothing for cas-pos, 3 runs for cas-cls and 4 runs for each of the other tasks
    assert len(jobs) == 3 + 4 * (len(ALL_TASKS) - 2)
    assert jobs.count("cd recipes/cas/scripts/ ; ./run_task_2.sh Dr-BERT/DrBERT-7GB") == 3
    assert not any("run_task_1.sh" in job for job in jobs if "recipes/cas/" in job)

    first, last = re.search(r"^#SBATCH --array=(\d+)-(\d+)%\d+", script, re.M).groups()
    assert (int(first), int(last)) == (0, len(jobs) - 1)


def test_write_slurm_script_with_another_number_of_runs(write_slurm_script):
    """`--nb-run` is the number of runs that each task must have."""
    completed = write_slurm_script({"cas-pos": 4, "cas-cls": 1, "clister-regr": 2}, "--nb-run", "2")
    assert completed.returncode == 0, completed.stderr[-3000:]

    jobs = re.findall(r'^    "(cd recipes/\w+/scripts/ ; \./run\S*\.sh .+)"$', completed.stdout, re.M)
    # Nothing for cas-pos and clister-regr, 1 run for cas-cls and 2 runs for each of the other tasks
    assert len(jobs) == 1 + 2 * (len(ALL_TASKS) - 3)
    assert jobs.count("cd recipes/cas/scripts/ ; ./run_task_2.sh Dr-BERT/DrBERT-7GB") == 1
    assert "cd recipes/cas/scripts/ ; ./run_task_1.sh Dr-BERT/DrBERT-7GB" not in jobs
    assert not any(job.startswith("cd recipes/clister/") for job in jobs)


def test_write_slurm_script_without_missing_run(write_slurm_script):
    completed = write_slurm_script({task.name: 4 for task in ALL_TASKS})
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "No missing run" in completed.stderr


def test_number_of_runs_is_checked(tmp_path):
    """`CheckNumberOfRuns.py` lists the scores that do not come from the expected number of runs: 4, or `--nb-run`."""
    repository = tmp_path / "repository"
    repository.mkdir()
    copy_tools(repository)
    (repository / "stats" / "results.json").write_text(json.dumps({
        "some/model": {
            "cas|pos|1.0": {"overall_f1": [0.5] * 4, "overall_accuracy": [0.5] * 4},
            "cas|cls|1.0": {"macro_f1": [0.5] * 2, "weighted_f1": [0.5] * 2},
        },
        "another/model": {"cas|pos|1.0": {"overall_f1": [0.5] * 5, "overall_accuracy": [0.5] * 5}},
    }))

    def listed(*arguments):
        completed = run_tool("CheckNumberOfRuns.py", repository, *arguments, cwd=tmp_path)
        assert completed.returncode == 0, completed.stderr[-3000:]
        # model  ->  dataset|task|fewshot  ->  metric  ->  number of runs
        return [tuple(line.split("  ->  ")) for line in completed.stdout.splitlines()]

    assert listed() == [
        ("some/model", "cas|cls|1.0", "macro_f1", "2"), ("some/model", "cas|cls|1.0", "weighted_f1", "2"),
        ("another/model", "cas|pos|1.0", "overall_f1", "5"), ("another/model", "cas|pos|1.0", "overall_accuracy", "5"),
    ]
    assert listed("--nb-run", "2") == [
        ("some/model", "cas|pos|1.0", "overall_f1", "4"), ("some/model", "cas|pos|1.0", "overall_accuracy", "4"),
        ("another/model", "cas|pos|1.0", "overall_f1", "5"), ("another/model", "cas|pos|1.0", "overall_accuracy", "5"),
    ]
    assert listed("--nb-run", "5") == [
        ("some/model", "cas|pos|1.0", "overall_f1", "4"), ("some/model", "cas|pos|1.0", "overall_accuracy", "4"),
        ("some/model", "cas|cls|1.0", "macro_f1", "2"), ("some/model", "cas|cls|1.0", "weighted_f1", "2"),
    ]


@pytest.mark.parametrize("tool", ["CheckNumberOfRuns.py", "WriteSlurmScriptForMissingRuns.py", "BuildExcelTableOfScores.py"])
def test_tools_expect_the_same_number_of_runs(tool):
    """The number of runs of the benchmark is written once, in `tools/number_of_runs.py`."""
    if tool == "WriteSlurmScriptForMissingRuns.py":
        pytest.importorskip("liquid", reason="python-liquid is needed by tools/WriteSlurmScriptForMissingRuns.py")
    nb_runs = load_module("drbenchmark_number_of_runs", REPO / "tools" / "number_of_runs.py").NB_RUNS
    usage = subprocess.run([sys.executable, str(REPO / "tools" / tool), "--help"], capture_output=True, text=True, timeout=120)
    assert usage.returncode == 0, usage.stderr[-3000:]
    assert f"(default: {nb_runs})" in " ".join(usage.stdout.split())


@pytest.mark.parametrize("arguments", [[], ["--jeanzay"]], ids=["slurm", "jean-zay"])
def test_generated_slurm_script_is_submitted_from_the_root_of_the_repository(arguments, write_slurm_script, tmp_path):
    """The jobs of the generated script are relative to the root of the repository: it refuses to run elsewhere."""
    script = tmp_path / "missing_runs.sh"
    script.write_text(write_slurm_script({"cas-pos": 4}, *arguments).stdout)

    refused, commands = run_without_slurm(script, cwd=tmp_path / "elsewhere", directory=tmp_path / "slurm")
    assert refused.returncode == 1
    assert "has to be submitted from the root of the DrBenchmark repository" in refused.stderr
    # Nothing was run: neither the set up of the environment, nor a job
    assert commands == []

    # From a directory with the recipes, the environment is set up and the job is run
    root = tmp_path / "root"
    (root / "recipes" / "cas" / "scripts").mkdir(parents=True)
    accepted, commands = run_without_slurm(script, cwd=root, directory=tmp_path / "slurm")
    assert "has to be submitted" not in accepted.stderr
    assert commands[-1] == "conda activate DrBenchmark"
    assert "index: 0, value: cd recipes/cas/scripts/ ; ./run_task_2.sh Dr-BERT/DrBERT-7GB" in accepted.stdout
