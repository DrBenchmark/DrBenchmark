"""Tests of the scripts for SLURM, run with bash instead of being submitted to SLURM.

These scripts run `run.py` in the directory they are submitted from, and SLURM writes their
logs in `./logs/`: they have to be submitted from the root of the repository, and say so
instead of running jobs that do nothing. What `run.py` does for a job is tested with it.
"""

import re

import pytest

from helpers import REPO, run_without_slurm

ALL_TASKS_SCRIPTS = ["run_all_slurm.sh", "run_all_jean_zay.sh"]
MODELS = ["first/model", "second/model"]


@pytest.fixture
def root(tmp_path):
    """A directory with the files the scripts look for at the root of the repository."""
    root = tmp_path / "root"
    (root / "recipes" / "cas" / "scripts").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "models.txt").write_text("\n".join(MODELS) + "\n")
    (root / "run.py").touch()
    return root


@pytest.mark.parametrize("script", ALL_TASKS_SCRIPTS + ["run_one_jean_zay.sh"])
def test_script_refuses_to_run_out_of_the_repository(script, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    completed, commands = run_without_slurm(REPO / script, cwd=elsewhere, directory=tmp_path / "slurm")
    assert completed.returncode == 1
    assert f"has to be submitted from the root of the DrBenchmark repository, not from {elsewhere}" in completed.stderr
    # Nothing was run: neither the set up of the environment, nor a job
    assert commands == []


@pytest.mark.parametrize("script", ALL_TASKS_SCRIPTS)
@pytest.mark.parametrize("missing", ["run.py", "models.txt", "recipes"])
def test_script_for_all_tasks_needs_the_files_of_the_repository(script, missing, root, tmp_path):
    """`run.py` is run in the directory the job is submitted from, where it reads `models.txt` and the recipes."""
    (root / missing).rename(tmp_path / missing)

    completed, commands = run_without_slurm(REPO / script, cwd=root, directory=tmp_path / "slurm")
    assert completed.returncode == 1
    assert "has to be submitted from the root of the DrBenchmark repository" in completed.stderr
    assert commands == []


@pytest.mark.parametrize("script", ALL_TASKS_SCRIPTS)
def test_script_for_all_tasks_asks_run_py_for_the_run_of_its_job(script, root, tmp_path):
    """Each job of the array does one run of every task for every model: the one that has its number for `run.py`."""
    runs = re.search(r"^NBR_RUNS=(\d+)$", (REPO / script).read_text(), re.M).group(1)

    for job in (0, 7):
        completed, commands = run_without_slurm(REPO / script, cwd=root, directory=tmp_path / "slurm", task_id=job)
        assert completed.returncode == 0, completed.stderr
        assert "conda activate DrBenchmark" in commands
        assert commands[-1] == f"srun python run.py --tasks all --nb-run {runs} --job {job}"


def test_script_for_one_task_calls_run_py(root, tmp_path):
    script = (REPO / "run_one_jean_zay.sh").read_text()
    task = re.search(r"^TASK='([^']+)'$", script, re.M).group(1)
    model = re.search(r'^MODEL_NAME="([^"]+)"$', script, re.M).group(1)
    runs = re.search(r"^NBR_RUNS=(\d+)$", script, re.M).group(1)

    completed, commands = run_without_slurm(REPO / "run_one_jean_zay.sh", cwd=root, directory=tmp_path / "slurm")
    assert completed.returncode == 0, completed.stderr
    assert commands[-1] == f"python run.py --tasks {task} --models {model} --nb-run {runs}"
