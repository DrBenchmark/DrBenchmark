"""What the tools analyse: the models of `models.txt` and the datasets of the benchmark."""

import os
from glob import glob

from repository_paths import MODELS_FILE, RECIPES_DIR, REPO_DIR


def models_to_evaluate():
    """Names of the models of `models.txt`: its lines, but the empty ones and the comments."""
    with open(MODELS_FILE) as f_in:
        return [line.strip() for line in f_in if line.strip() and not line.startswith('#')]


def local_model(name, offline):
    """What is given to `from_pretrained` to load a model.

    In offline mode, a model that is not given by a path is read where
    `download_models_locally.py` saves it, as the recipes do.
    """
    if offline and not os.path.exists(name):
        return os.path.join(REPO_DIR, "models", name.lower().replace('/', '_'))
    return name


def load_benchmark_dataset(name, subset, offline, **kwargs):
    """A dataset of the benchmark, with its three splits: `DrBenchmark/<corpus>` on the HuggingFace Hub.

    In offline mode, it is read where `download_datasets_locally.py` saves it, as the
    recipes do: `recipes/<corpus>/data/local_hf_<subset>/`. `kwargs` are given to
    `datasets.load_dataset`.
    """
    # Imported here: most of the tools do not read any dataset
    from datasets import load_dataset, load_from_disk

    if not offline:
        return load_dataset(name, subset, **kwargs)

    data_dir = os.path.join(RECIPES_DIR, name.split('/')[1].lower(), "data")
    path = os.path.join(data_dir, f"local_hf_{subset}")
    if subset is None and not os.path.isdir(path):
        # The only subset of the dataset, which the Hub gives when none is asked for
        copies = glob(os.path.join(data_dir, "local_hf_*"))
        if len(copies) == 1:
            path = copies[0]
    return load_from_disk(path)


def label_names(feature):
    """Names of the labels of a column of labels, or of lists of labels.

    A list of labels is a `Sequence` or a `List` for `datasets`, depending on its version,
    and was a Python list for the oldest ones.
    """
    if isinstance(feature, list):
        feature = feature[0]
    return getattr(feature, "feature", feature).names
