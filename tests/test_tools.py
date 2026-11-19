"""Tests of the tools that aggregate the results of the runs (`tools/`)."""

import shutil
import subprocess
import sys

import pytest

from helpers import ALL_TASKS, REPO, read_json

pytestmark = pytest.mark.e2e

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


def test_compute_metrics(results_directory, tiny_bert):
    completed = run_tool("ComputeMetrics.py", results_directory)
    assert completed.returncode == 0, completed.stderr[-3000:]

    results = read_json(results_directory / "stats" / "results.json")
    averages = read_json(results_directory / "stats" / "overall_averaged_metrics.json")
    assert list(results) == list(averages) == [str(tiny_bert)]

    expected = {
        f"{task.recipe}|{task.name.split('-')[1]}|1.0": COLLECTED_METRICS[task.kind] for task in ALL_TASKS
    }
    for scores in (results[str(tiny_bert)], averages[str(tiny_bert)]):
        assert {key: set(metrics) for key, metrics in scores.items()} == expected
    # One run per task
    assert all(len(values) == 1 for metrics in results[str(tiny_bert)].values() for values in metrics.values())


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
