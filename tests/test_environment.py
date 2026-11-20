"""Tests of what a run writes about the versions that computed it: the libraries and the code of the repository."""

import importlib.metadata
import platform
import shutil
import subprocess

import pytest


def test_versions_of_the_libraries(training):
    environment = training.environment()
    assert environment["python"] == platform.python_version()
    assert set(training.LIBRARIES) <= set(environment)
    for library in training.LIBRARIES:
        assert environment[library] == importlib.metadata.version(library)
    # A library that is not installed has no version
    assert {"torch", "transformers", "datasets", "scikit-learn"} <= set(training.LIBRARIES)
    assert set(environment) == {"python", *training.LIBRARIES, "commit", "modified"}


def test_code_which_is_not_in_a_git_repository_has_no_commit(training, monkeypatch, tmp_path):
    """The archive of the repository, for instance: its commit is not known."""
    monkeypatch.setattr(training.utils, "REPO_DIR", str(tmp_path))
    assert training.code_version() == {"commit": None, "modified": None}


@pytest.fixture
def repository(training, monkeypatch, tmp_path):
    """A git repository with one commit, which stands for the one of the scripts: its directory and its commit."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    directory = tmp_path.resolve() / "repository"
    directory.mkdir()

    def git(*arguments):
        identity = ["-c", "user.name=Somebody", "-c", "user.email=somebody@example.org", "-c", "commit.gpgsign=false"]
        completed = subprocess.run(["git", "-C", str(directory), *identity, *arguments], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init", "--quiet")
    (directory / "script.py").write_text("first version\n")
    git("add", "script.py")
    git("commit", "--quiet", "-m", "First version")
    monkeypatch.setattr(training.utils, "REPO_DIR", str(directory))
    return directory, git("rev-parse", "HEAD")


def test_commit_of_the_repository(training, repository):
    repository, commit = repository
    assert len(commit) == 40
    assert training.code_version() == {"commit": commit, "modified": False}

    # The results of the runs and the models are not files of the repository
    (repository / "results.json").write_text("{}")
    assert training.code_version() == {"commit": commit, "modified": False}

    # A script that was changed since the commit: the run was not computed by the code of the commit
    (repository / "script.py").write_text("second version\n")
    assert training.code_version() == {"commit": commit, "modified": True}


def test_copy_of_the_scripts_in_another_repository_has_no_commit(training, repository, monkeypatch):
    """The commit of a repository that contains a copy of the scripts is not the one of the scripts."""
    copy = repository[0] / "somewhere" / "DrBenchmark"
    copy.mkdir(parents=True)
    monkeypatch.setattr(training.utils, "REPO_DIR", str(copy))
    assert training.code_version() == {"commit": None, "modified": None}
