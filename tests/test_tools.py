"""Tests of the tools that work on the results of the runs (`tools/`)."""

import json
import re
import shutil
import subprocess
import sys

import pytest

from helpers import ALL_TASKS, REPO, read_json

# Metrics collected by `tools/ComputeMetrics.py` for each kind of task
COLLECTED_METRICS = {
    "pos": {"overall_f1", "overall_accuracy"},
    "ner": {"overall_f1", "overall_accuracy"},
    "cls": {"macro_f1", "weighted_f1"},
    "mcqa": {"hamming_score", "exact_match"},
    "regr": {"edrm", "spearman_correlation_coef"},
}


def run_tool(script, directory, *arguments):
    """Run a tool from `directory`, as the tools read and write relative paths."""
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / script), *arguments],
        cwd=directory, capture_output=True, text=True, timeout=300,
    )


def results_key(task):
    """Key of a task in `stats/results.json`, for the runs on the whole training set."""
    return f"{task.recipe}|{task.name.split('-')[1]}|1.0"


@pytest.fixture(scope="module")
def results_directory(run_recipe, tiny_bert, tmp_path_factory):
    """A copy of the layout of the repository, with one run of every task."""
    directory = tmp_path_factory.mktemp("results")
    (directory / "stats").mkdir()
    for task in ALL_TASKS:
        run = run_recipe(task.name, tiny_bert)
        run.check()
        runs_directory = directory / "recipes" / task.recipe / "runs"
        runs_directory.mkdir(parents=True, exist_ok=True)
        for run_file in run.run_files:
            shutil.copy(run_file, runs_directory)
    return directory


@pytest.mark.e2e
def test_compute_metrics(results_directory, tiny_bert):
    completed = run_tool("ComputeMetrics.py", results_directory)
    assert completed.returncode == 0, completed.stderr[-3000:]

    results = read_json(results_directory / "stats" / "results.json")
    averages = read_json(results_directory / "stats" / "overall_averaged_metrics.json")
    assert list(results) == list(averages) == [str(tiny_bert)]

    expected = {results_key(task): COLLECTED_METRICS[task.kind] for task in ALL_TASKS}
    for scores in (results[str(tiny_bert)], averages[str(tiny_bert)]):
        assert {key: set(metrics) for key, metrics in scores.items()} == expected
    # One run per task
    assert all(len(values) == 1 for metrics in results[str(tiny_bert)].values() for values in metrics.values())


@pytest.mark.e2e
def test_results_to_benchmark(results_directory):
    assert run_tool("ComputeMetrics.py", results_directory).returncode == 0
    completed = run_tool("ResultsToBenchmark.py", results_directory, "--nb-run", "1", "--csv")
    assert completed.returncode == 0, completed.stderr[-3000:]

    lines = completed.stdout.strip().splitlines()
    assert lines[0].startswith("dataset,task,metric")
    # One line per score shown on the benchmark: every task but DEFT-2019,
    # and two scores for the question answering and similarity tasks
    two_scores = sum(task.kind in ("mcqa", "regr") for task in ALL_TASKS)
    assert len(lines) - 1 == len(ALL_TASKS) - 1 + two_scores


@pytest.fixture
def slurm_from_missing_runs(tmp_path):
    """Run `tools/SlurmFromMissingRuns.py` on the number of runs given for each task."""
    pytest.importorskip("liquid", reason="python-liquid is needed by tools/SlurmFromMissingRuns.py")

    def run(runs_per_task):
        (tmp_path / "stats").mkdir(exist_ok=True)
        results = {
            results_key(task): {"overall_f1": [0.5] * runs_per_task[task.name]}
            for task in ALL_TASKS if runs_per_task.get(task.name)
        }
        (tmp_path / "stats" / "results.json").write_text(json.dumps({"Dr-BERT/DrBERT-7GB": results}))
        return run_tool("SlurmFromMissingRuns.py", tmp_path)

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
