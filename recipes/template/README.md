# MyCorpus

This directory is the template of a recipe: [ADDING_A_TASK.md](../ADDING_A_TASK.md) says how to make a task of the benchmark of it. This file is the README of the recipe, to fill in.

TODO: what the corpus is, where it comes from and how to cite it.

## Data

TODO: where the corpus is hosted, e.g. on the HuggingFace Hub ([DrBenchmark/MyCorpus](https://huggingface.co/datasets/DrBenchmark/MyCorpus)), and the size of its splits.

| Train | Validation | Test |
|------:|-----------:|-----:|
|  TODO |       TODO | TODO |

The corpus is downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, it is read from `./data/local_hf_None/`: this directory is created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

From the root of the repository:

```bash
python run.py --tasks mycorpus-cls --models "Dr-BERT/DrBERT-7GB"
```

Or with the launcher of the task, which can be run from any directory:

```bash
bash recipes/mycorpus/scripts/run.sh "Dr-BERT/DrBERT-7GB"
```

## Classification

TODO: what a model has to predict for a text, and the classes.

- **Launcher**: `run.sh` (configuration: `yaml/cls.yaml`)
- **Metric**: weighted F1
