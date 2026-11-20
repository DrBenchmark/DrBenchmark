# Overview

Each sub-directory contains the recipe of one dataset of DrBenchmark:

- `README.md`: description of the dataset and of its tasks
- `scripts/`: fine-tuning and evaluation scripts, with one launcher (`run*.sh`) per task. A script only states what is specific to its task: the code shared by the recipes is in [common/](./common/), which each script finds from its own location
- `yaml/`: default hyper-parameters of each task
- `data/`: local copy of the data (only used in offline mode and for DEFT 2019)
- `runs/`: results of the runs (one JSON file per run)
- `save_models/`: best model of each run

## Datasets

| Dataset                           | Task                                                          | Task name for `run.py`  | Launcher (from `scripts/`)         |
|:----------------------------------|:--------------------------------------------------------------|:------------------------|:-----------------------------------|
| [CAS](./cas/)                     | Task 1 - Part-Of-Speech Tagging (POS)                         | `cas-pos`               | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-class Classification (CLS)                     | `cas-cls`               | `./run_task_2.sh <model>`          |
|                                   | Task 3 - Named Entity Recognition (NER), negation             | `cas-ner_neg`           | `./run_task_3.sh <model>`          |
|                                   | Task 4 - Named Entity Recognition (NER), speculation          | `cas-ner_spec`          | `./run_task_4.sh <model>`          |
| [ESSAI](./essai/)                 | Task 1 - Part-Of-Speech Tagging (POS)                         | `essai-pos`             | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-class Classification (CLS)                     | `essai-cls`             | `./run_task_2.sh <model>`          |
|                                   | Task 3 - Named Entity Recognition (NER), negation             | `essai-ner_neg`         | `./run_task_3.sh <model>`          |
|                                   | Task 4 - Named Entity Recognition (NER), speculation          | `essai-ner_spec`        | `./run_task_4.sh <model>`          |
| [QUAERO](./quaero/)               | Named Entity Recognition (NER), EMEA                          | `quaero-ner_emea`       | `./run.sh <model> emea`            |
|                                   | Named Entity Recognition (NER), MEDLINE                       | `quaero-ner_medline`    | `./run.sh <model> medline`         |
| [E3C](./e3c/)                     | Task 1 - Named Entity Recognition (NER), clinical entities    | `e3c-ner_clinical`      | `./run.sh <model> French_clinical` |
|                                   | Task 2 - Named Entity Recognition (NER), temporal information | `e3c-ner_temporal`      | `./run.sh <model> French_temporal` |
| [MorFITT](./morfitt/)             | Multi-label Classification (CLS)                              | `morfitt-cls`           | `./run.sh <model>`                 |
| [FrenchMedMCQA](./frenchmedmcqa/) | Task 1 - Multiple-Choice Question Answering (MCQA)            | `frenchmedmcqa-mcqa`    | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-class Classification (CLS)                     | `frenchmedmcqa-cls`     | `./run_task_2.sh <model>`          |
| [MantraGSC](./mantragsc/)         | Named Entity Recognition (NER), EMEA                          | `mantragsc-ner_emea`    | `./run.sh <model> fr_emea`         |
|                                   | Named Entity Recognition (NER), MEDLINE                       | `mantragsc-ner_medline` | `./run.sh <model> fr_medline`      |
|                                   | Named Entity Recognition (NER), Patents                       | `mantragsc-ner_patents` | `./run.sh <model> fr_patents`      |
| [CLISTER](./clister/)             | Semantic Textual Similarity (STS)                             | `clister-regr`          | `./run.sh <model>`                 |
| [DEFT 2019](./deft2019/)          | Named Entity Recognition (NER)                                | `deft2019-ner`          | `./run.sh <model>`                 |
| [DEFT 2020](./deft2020/)          | Task 1 - Semantic Textual Similarity (STS)                    | `deft2020-regr`         | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-class Classification (CLS)                     | `deft2020-cls`          | `./run_task_2.sh <model>`          |
| [DEFT 2021](./deft2021/)          | Task 1 - Named Entity Recognition (NER)                       | `deft2021-ner`          | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-label Classification (CLS)                     | `deft2021-cls`          | `./run_task_2.sh <model>`          |
| [DiaMED](./diamed/)               | Multi-class Classification (CLS)                              | `diamed-cls`            | `./run.sh <model>`                 |
| [PxCorpus](./pxcorpus/)           | Task 1 - Named Entity Recognition (NER)                       | `pxcorpus-ner`          | `./run_task_1.sh <model>`          |
|                                   | Task 2 - Multi-class Classification (CLS)                     | `pxcorpus-cls`          | `./run_task_2.sh <model>`          |

The tasks 2, 3 and 4 of CAS and ESSAI, and DEFT 2019, are not part of the 20 tasks evaluated in the [DrBenchmark paper](https://hal.science/hal-04470938).

## Running a recipe

From the root of the repository, with `run.py`:

```bash
python run.py --tasks cas-pos --models "Dr-BERT/DrBERT-7GB"
```

`run.py` gives each run its seed (0, then 1...) and the options of the recipes it is given, e.g. `--epochs 5` (see the [main README](../README.md)).

Or directly, with the Python script of the task and its YAML file: this is what `run.py` starts for the task, without bash.

```bash
python recipes/cas/scripts/task_1_finetuning_bert_pos.py --config recipes/cas/yaml/pos.yaml --model_name "Dr-BERT/DrBERT-7GB"
```

The launcher of the task, in the `scripts/` directory of its recipe, is a shortcut for a shell: it starts the same script with the same YAML file. The scripts and the launchers find their files from their own location: they can be run from any directory.

```bash
bash recipes/cas/scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB"
```

The other arguments override the values of the YAML file: they follow the model for a launcher (and the subset, for the launchers that take one), e.g. `bash recipes/cas/scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB" --seed 42 --epochs 5`. A relative path given on the command line (`--data_dir`, `--run_dir`, `--output_dir` or a local model) is relative to the current directory. In the results of the run, it is written from the `scripts/` directory, as the paths of the YAML file are: the name of a run does not depend on the directory it is started from.

Each run writes its metrics, its hyper-parameters, its predictions on the test set and the versions that computed it (Python, the libraries and the commit of the repository) in `runs/DrBenchmark-<task name>-<hash>.json`, and saves its best model in `save_models/DrBenchmark-<task name>-<hash>_best_model/`. The hash is the one of the arguments of the run, among which its seed. With `--seed`, a run whose results exist is not computed again; without it, a seed that was not used yet is drawn.

The best model is saved with the names of its classes, with a description (`drbenchmark.json`) and with a model card (`README.md`): `predict.py` predicts with it on other data (see [Using a trained model](../README.md#using-a-trained-model)). `--keep_model false` does not keep it: the run is the same one, with the same name and the same results.

## The script of a task

The Python script of a task only states what is specific to the task, in a line `TASK = ...`, and trains a model when it is started (`TASK.train()`). The rest is in [common/](./common/), which has one module for each kind of task: `token_classification`, `sequence_classification`, `multi_label_classification` and `regression`.

```python
def build_text(e, tokenizer):
    return f"{' '.join(e['tokens'])}"


TASK = sequence_classification.Task(dataset="DrBenchmark/CAS", build_text=build_text, inputs={"tokens": "words"})

if __name__ == '__main__':
    TASK.train()
```

A script reads the arguments of its run first, with `utils.stop_if_already_computed()`, before it imports the module of its kind of task: this import loads the libraries that train, which takes seconds, and a run whose results exist stops before.

`predict.py` loads the same script to predict with a model that a run kept: what the model reads is built by the same code, `TASK.encode`, when the model is trained and when it predicts. A script that builds its text itself, as the one above, also says which columns of an example it reads (`inputs`: a `"text"`, or the `"words"` of a sentence): they are what `predict.py` asks for.

## Data

Except for DEFT 2019, whose access is restricted (see [its README](./deft2019/)), the datasets are hosted on the [HuggingFace Hub](https://huggingface.co/DrBenchmark). The way they are loaded depends on the value of `offline` in `config.yaml` (see the [main README](../README.md)):

- `offline: False`: the datasets and the models are downloaded automatically from the HuggingFace Hub.
- `offline: True`, which is the default value of the repository: the datasets are read from the `data/local_hf_<subset>/` directory of each recipe and the models from the `models/` directory of the repository. They have to be created beforehand with `python download_datasets_locally.py` and `python download_models_locally.py`, run from the root of the repository.

The value of `config.yaml` can be overridden for one run by giving `--offline true` or `--offline false` to a launcher or to `run.py`.
