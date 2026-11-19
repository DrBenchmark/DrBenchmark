"""Tests of the tools that work on the results of the runs (`tools/`)."""

import ast
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from helpers import ALL_TASKS, REPO, read_json, run_without_slurm

TOOLS = sorted(path.name for path in (REPO / "tools").glob("*.py"))

# Metrics collected by `tools/ComputeMetrics.py` for each kind of task
COLLECTED_METRICS = {
    "pos": {"overall_f1", "overall_accuracy"},
    "ner": {"overall_f1", "overall_accuracy"},
    "cls": {"macro_f1", "weighted_f1"},
    "mcqa": {"hamming_score", "exact_match"},
    "regr": {"edrm", "spearman_correlation_coef"},
}


def copy_tools(repository):
    """Make `repository` a directory where the tools work as in the real one.

    The tools find `stats/`, `recipes/` and `models.txt` from their own location: their
    copies use the ones of `repository`.
    """
    shutil.copytree(REPO / "tools", repository / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    # `tools/SlurmFromMissingRuns.py` reads the tasks in `run.py`
    shutil.copy(REPO / "run.py", repository)
    (repository / "stats").mkdir()


def run_tool(script, repository, *arguments, cwd):
    """Run the copy of a tool which is in `repository`, from the directory `cwd`."""
    return subprocess.run(
        [sys.executable, str(repository / "tools" / script), *arguments],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )


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
    """The tools find the files of the repository with `tools/paths.py`: they can be run from any directory."""
    relative_paths = []
    for node in ast.walk(ast.parse((REPO / "tools" / tool).read_text())):
        if isinstance(node, ast.Call) and node.args:
            function = getattr(node.func, "id", getattr(node.func, "attr", None))
            path = beginning_of_literal(node.args[0])
            if function in PATH_FUNCTIONS and path is not None and not os.path.isabs(path):
                relative_paths.append(f"line {node.lineno}: {function}({path!r}")
    assert not relative_paths


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
def test_compute_metrics(results_repository, tiny_bert, tmp_path):
    # The tool is run from another directory: it must find the results in its repository
    completed = run_tool("ComputeMetrics.py", results_repository, cwd=tmp_path)
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
def test_results_to_benchmark(results_repository, tmp_path):
    assert run_tool("ComputeMetrics.py", results_repository, cwd=tmp_path).returncode == 0
    completed = run_tool("ResultsToBenchmark.py", results_repository, "--nb-run", "1", "--csv", cwd=tmp_path)
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
        "ResultsToBenchmark.py", results_repository, "--nb-run", "1", "--csv", "--results", "copy.json", cwd=tmp_path,
    )
    assert with_path.returncode == 0, with_path.stderr[-3000:]
    assert with_path.stdout == completed.stdout


@pytest.fixture
def slurm_from_missing_runs(tmp_path):
    """Run `tools/SlurmFromMissingRuns.py` on the number of runs given for each task."""
    pytest.importorskip("liquid", reason="python-liquid is needed by tools/SlurmFromMissingRuns.py")
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
        return run_tool("SlurmFromMissingRuns.py", repository, *arguments, cwd=elsewhere)

    return run


def test_slurm_from_missing_runs(slurm_from_missing_runs):
    """The generated job array has one index per missing run, for 4 runs per task."""
    completed = slurm_from_missing_runs({"cas-pos": 4, "cas-cls": 1})
    assert completed.returncode == 0, completed.stderr[-3000:]
    script = completed.stdout

    jobs = re.findall(r'^    "(cd recipes/\w+/scripts/ ; \./run\S*\.sh .+)"$', script, re.M)
    # Nothing for cas-pos, 3 runs for cas-cls and 4 runs for each of the other tasks
    assert len(jobs) == 3 + 4 * (len(ALL_TASKS) - 2)
    assert jobs.count("cd recipes/cas/scripts/ ; ./run_task_2.sh Dr-BERT/DrBERT-7GB") == 3
    assert not any("run_task_1.sh" in job for job in jobs if "recipes/cas/" in job)

    first, last = re.search(r"^#SBATCH --array=(\d+)-(\d+)%\d+", script, re.M).groups()
    assert (int(first), int(last)) == (0, len(jobs) - 1)


def test_slurm_from_missing_runs_without_missing_run(slurm_from_missing_runs):
    completed = slurm_from_missing_runs({task.name: 4 for task in ALL_TASKS})
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "No missing run" in completed.stderr


@pytest.mark.parametrize("arguments", [[], ["--jeanzay"]], ids=["slurm", "jean-zay"])
def test_generated_slurm_script_is_submitted_from_the_root_of_the_repository(arguments, slurm_from_missing_runs, tmp_path):
    """The jobs of the generated script are relative to the root of the repository: it refuses to run elsewhere."""
    script = tmp_path / "missing_runs.sh"
    script.write_text(slurm_from_missing_runs({"cas-pos": 4}, *arguments).stdout)

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
