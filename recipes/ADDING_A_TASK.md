# Adding a task

A task of the benchmark is a Python script and a YAML file in a recipe, with a line in `run.py`. This guide adds the classification task of a corpus called MyCorpus, from the recipe in [template/](./template/): the same steps add a task of another kind.

```bash
cp -r recipes/template recipes/mycorpus
```

| File of the recipe                | What it is                                                                         |
|:----------------------------------|:-----------------------------------------------------------------------------------|
| `scripts/finetuning_bert_cls.py`  | The script of the task: it states what is specific to the task, in `TASK = ...`    |
| `yaml/cls.yaml`                   | Its hyper-parameters, its name and its dataset                                     |
| `scripts/run.sh`                  | Its launcher, a shortcut for a shell: the script with its YAML file                |
| `README.md`                       | The description of the corpus and of its tasks                                     |
| `data/`, `runs/`, `save_models/`  | Where the data are read in offline mode, and where the runs write their results and their models. Their `.gitignore` keeps these directories in the repository, without what is written in them: a run does not create `runs/` |

What is to change in the first four files is marked `TODO`.

## 1. Choose the kind of the task

The code that trains and evaluates is in [common/](./common/), with one module for each kind of task. The script of a task creates the `Task` of its kind.

| Kind of task                         | Module                         | What the script gives                                                   | Columns of the dataset                                                                                         | Its name has    |
|:-------------------------------------|:-------------------------------|:------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:----------------|
| A label for each word (POS, NER)     | `token_classification`         | the column of the tags (`ner_tags` by default), the score that chooses the checkpoint | `id`, `tokens` (the words of a sentence) and the tags: the classes of the words                  | `pos` or `ner`  |
| One class for a text                 | `sequence_classification`      | `build_text`, the columns it reads (`inputs`), the column of the class (`label` by default) | an identifier (`id` by default), the class, and what `build_text` reads                      | `cls`           |
| Several labels for a text            | `multi_label_classification`   | the column of the text                                                  | `id`, the text, `specialities` (the labels of the text) and `specialities_one_hot` (0 or 1 for each label)     | `cls`           |
| A similarity score for two sentences | `regression`                   | the two columns of the sentences, the column of the score               | `id`, the two sentences and a score between 0 and 5                                                            | `regr`          |

- A dataset has three splits: `train`, `validation` and `test`. Its classes are `ClassLabel` features: their names are the ones a model is saved with.
- **The name of a task says its kind.** A name is `<recipe>-<task>`, with one hyphen: `mycorpus-cls`. The recipe is the name of its directory. The tools choose the scores of a task from what follows the hyphen (`scores_of` in [tools/CollectAndAverageScores.py](../tools/CollectAndAverageScores.py)): it has to have `pos`, `ner`, `cls`, `regr` or `mcqa`.
- A task of another kind has a class of its own in its script, as the question answering task of [FrenchMedMCQA](./frenchmedmcqa/scripts/task_1_finetuning_bert_mcqa.py), which changes the classes and the scores of a classification.

## 2. Write the script and the YAML file

In `scripts/finetuning_bert_cls.py`, the task of the template is the classification of a text:

```python
def build_text(e, tokenizer):
    return e["text"]


TASK = sequence_classification.Task(dataset="DrBenchmark/MyCorpus", build_text=build_text, inputs={"text": "text"}, label_column="label")
```

- `dataset` is the name of the dataset on the HuggingFace Hub.
- `build_text` returns the text that the model reads for an example. `inputs` names the columns that it reads, and says whether each is a `"text"` or the `"words"` of a sentence: `predict.py` asks for them, and builds the text with the same function.
- The first lines of the script are the ones of every script: they find the shared code, and read the arguments of the run before the libraries that train are loaded.

In `yaml/cls.yaml`, `task` is the name of the task, `dataset` the same dataset, and `subset` the subset that is read when the dataset has several. The hyper-parameters are the values of a run: an option of the command line takes the place of one of them (`--epochs 5`). The three directories are the ones of every recipe.

The launcher and the README name these files: change them when the files are renamed.

## 3. Add the task to `run.py`

A line of `task2script`, in [run.py](../run.py), with the script and the YAML file:

```python
    'mycorpus-cls': Script('finetuning_bert_cls.py', 'cls.yaml'),
```

One script can do several tasks, each with a YAML file of its own, as the two scope detection tasks of CAS. A script that reads several subsets of a dataset with one YAML file has the subset as a third value, as E3C: `utils.subset2task_detail`, in [common/utils.py](./common/utils.py), then gives what the subset adds to the name of the task.

## 4. Give the data

- With `offline: False` in `config.yaml`, or `--offline false`, the dataset is downloaded from the HuggingFace Hub.
- With `offline: True`, the default value, it is read from `recipes/mycorpus/data/local_hf_<subset>/`: `local_hf_None` for a dataset without subset. [download_datasets_locally.py](../download_datasets_locally.py) saves there the datasets of its list `ds`, which are the `DrBenchmark/<Corpus>` of the Hub: add `["MyCorpus", None]` to it. A dataset from somewhere else is saved by hand, with `save_to_disk` of `datasets`.

## 5. Try the task

```bash
python run.py --tasks mycorpus-cls --models Dr-BERT/DrBERT-7GB --epochs 1
python tools/CollectAndAverageScores.py
python predict.py --task mycorpus-cls --model Dr-BERT/DrBERT-7GB "Le patient présente une toux sèche."
```

The first command trains and evaluates a model, and writes its results in `recipes/mycorpus/runs/`. The second one collects the scores in `stats/results.json`. The third one predicts the class of a sentence with the model that the run kept. [TUTORIAL.md](../TUTORIAL.md) shows what these commands print, for a task of the benchmark.

## 6. Add it to the tests

The tests run every task of `run.py` with a tiny model, on synthetic datasets that have the columns of the real ones: add the one of the recipe to `build_all`, in [tests/synthetic.py](../tests/synthetic.py), with a key for each of its subsets (`None` without subset).

```python
        "mycorpus": {
            None: build(
                {"id": str, "text": text, "label": lambda i: i % 3},
                {"id": Value("string"), "text": Value("string"), "label": ClassLabel(names=["first", "second", "third"])},
            ),
        },
```

`pytest` then trains the task, predicts with its model and gives its results to the tools. `pytest -m "not e2e"` only does the fast checks, which say what is missing for a task of the benchmark:

- the YAML file has the keys of the other ones, and its dataset is `DrBenchmark/<Corpus>`, for a recipe named `<corpus>`;
- the README of the recipe names the task, its launcher (`` `run.sh` ``), its YAML file (`` `yaml/cls.yaml` ``) and its subset;
- the table of [README.md](./README.md) has a row for the task, with the command of its launcher;
- every script, launcher and YAML file of the recipe belongs to a task;
- `download_datasets_locally.py` saves its dataset;
- the job arrays of `run_all_slurm.sh` and `run_all_jean_zay.sh` have one job per run: `--array` is the number of tasks x the number of models x the number of runs.

## 7. Add it to the tables

- [tools/BuildExcelTableOfScores.py](../tools/BuildExcelTableOfScores.py) lists the rows of the table of the benchmark: add the task to `order`, and its corpus to `dataset2pretty`.
- The main [README.md](../README.md) has the table of the tasks and a description of each dataset.
- The tools that analyse the datasets, [CountClassesOfDatasets.py](../tools/CountClassesOfDatasets.py) and [CountSubwordsPerWord.py](../tools/CountSubwordsPerWord.py), have their own lists of datasets.
