"""Tests of the scripts that save the datasets and the models for the offline mode.

Nothing is downloaded: the functions that download are replaced by ones that record
where the datasets and the models are saved. It has to be where the recipes read them,
wherever the scripts are run from.
"""

import os
import sys
import types

import pytest

from helpers import ALL_TASKS, MODELS_FILE, MODELS_OF_THE_FILE, RECIPES, REPO, Task, load_module


class Saved:
    """Stands for a dataset, a model or a tokenizer: records the directories it is saved to."""

    def __init__(self):
        self.directories = []

    def save_to_disk(self, directory):
        self.directories.append(os.path.realpath(directory))

    save_pretrained = save_to_disk


def task_ids(task):
    return task.name


# DEFT-2019 is not downloaded: its access is restricted and it is not on the Hub
@pytest.mark.parametrize("task", [task for task in ALL_TASKS if task.recipe != "deft2019"], ids=task_ids)
def test_dataset_is_saved_where_the_recipe_reads_it(task, monkeypatch, tmp_path):
    module = load_module("download_datasets_locally", REPO / "download_datasets_locally.py")
    saved = Saved()
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: saved)
    monkeypatch.chdir(tmp_path)

    (corpus,) = {corpus for corpus, subset in module.ds if corpus.lower() == task.recipe}
    module.save_locally([corpus, task.subset])

    # `common/training.py` reads `<data_dir>/local_hf_<subset>/`, and `data_dir` is relative to `scripts/`
    read = task.scripts_dir / task.config["data_dir"] / f"local_hf_{task.subset}"
    assert saved.directories == [os.path.realpath(read)]
    assert saved.directories[0].startswith(str(RECIPES / task.recipe / "data"))
    assert not list(tmp_path.iterdir())


def test_models_to_download_are_the_ones_of_the_models_file(monkeypatch, tmp_path):
    """Neither the empty lines nor the comments of `models.txt` are models to download."""
    module = load_module("download_models_locally", REPO / "download_models_locally.py")
    (tmp_path / "models.txt").write_text(MODELS_FILE)
    monkeypatch.setattr(module, "REPO_DIR", str(tmp_path))
    assert module.models_to_download() == MODELS_OF_THE_FILE


@pytest.fixture
def download_models(monkeypatch, tmp_path):
    """Run `download_models_locally.py` with arguments, in a repository that has a `models.txt`: the models it saves."""
    module = load_module("download_models_locally", REPO / "download_models_locally.py")
    (tmp_path / "models.txt").write_text(MODELS_FILE)
    monkeypatch.setattr(module, "REPO_DIR", str(tmp_path))
    saved = []
    monkeypatch.setattr(module, "save_locally", saved.append)

    def download(*arguments):
        saved.clear()
        monkeypatch.setattr(sys, "argv", ["download_models_locally.py", *arguments])
        module.main()
        return list(saved)

    return download


def test_models_of_the_models_file_are_saved_by_default(download_models):
    assert download_models() == MODELS_OF_THE_FILE


def test_models_given_on_the_command_line_are_saved(download_models):
    """`--models`: other models than the ones of `models.txt`, as for `run.py`."""
    assert download_models("--models", "Dr-BERT/DrBERT-7GB") == ["Dr-BERT/DrBERT-7GB"]
    assert download_models("--models", "another/model", "first/model") == ["another/model", "first/model"]


def test_models_are_needed(download_models, tmp_path, capsys):
    """Without `models.txt`, the script says how to give the models instead of failing on the file."""
    (tmp_path / "models.txt").unlink()
    with pytest.raises(SystemExit) as stopped:
        download_models()
    assert stopped.value.code == f"`models.txt` is not in {tmp_path}: give the models to save with --models."
    # The file is not needed when the models are given
    assert download_models("--models", "another/model") == ["another/model"]

    with pytest.raises(SystemExit) as stopped:
        download_models("--models")
    assert stopped.value.code == 2 and "argument --models: expected at least one argument" in capsys.readouterr().err


def test_model_is_saved_where_the_recipes_read_it(utils, monkeypatch, tmp_path):
    module = load_module("download_models_locally", REPO / "download_models_locally.py")
    saved = Saved()
    for name in ("AutoModel", "AutoTokenizer"):
        monkeypatch.setattr(module, name, types.SimpleNamespace(from_pretrained=lambda model_name: saved))
    monkeypatch.chdir(tmp_path)

    module.save_locally("Dr-BERT/DrBERT-7GB")
    assert saved.directories == [str(REPO / "models" / "dr-bert_drbert-7gb")] * 2
    assert not list(tmp_path.iterdir())

    # In offline mode, the scripts of the recipes read the model there
    task = Task("cas-pos")
    monkeypatch.chdir(task.scripts_dir)
    monkeypatch.setattr(sys, "argv", [task.script, *task.script_args("Dr-BERT/DrBERT-7GB"), "--offline", "True"])
    read = utils.parse_args().model_name
    assert os.path.realpath(task.scripts_dir / read) == saved.directories[0]
