# Tools

Scripts that work on what the benchmark produces and uses: the results of the runs, the models and the datasets. They are not needed to run the benchmark itself (see the [main README](../README.md)).

```bash
python tools/CollectAndAverageScores.py
python tools/CollectAndAverageScores.py --help
```

- They can be run from any directory: they read and write in the repository, whose files they find from their own location (`recipes/<recipe>/runs/`, `stats/`, `models.txt`).
- A tool has the name of what it does. The files in lower case are not tools, but the [code they share](#shared-code).
- Every tool describes itself and its options with `--help`.
- Their dependencies are in `requirements.txt` (`openpyxl`, `seaborn` and `python-liquid` are only used here).

| Tool                                                                     | What it does                                                              | Writes                                                                                        |
|:-------------------------------------------------------------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| [CollectAndAverageScores.py](./CollectAndAverageScores.py)               | Collects and averages the scores of the runs                              | `stats/results.json`, `stats/overall_averaged_metrics.json`, `stats/overall_std_metrics.json` |
| [CheckNumberOfRuns.py](./CheckNumberOfRuns.py)                           | Lists the tasks that do not have the expected number of runs              | printed                                                                                       |
| [WriteSlurmScriptForMissingRuns.py](./WriteSlurmScriptForMissingRuns.py) | Writes a script for SLURM that does the missing runs                      | printed                                                                                       |
| [BuildExcelTableOfScores.py](./BuildExcelTableOfScores.py)               | Builds the table of the benchmark                                         | `stats/overall_results_<n>runs.xlsx`, or CSV printed                                          |
| [BuildLatexTableOfScores.py](./BuildLatexTableOfScores.py)               | Builds the LaTeX table of the scores                                      | printed                                                                                       |
| [PlotFewShotScores.py](./PlotFewShotScores.py)                           | Draws the scores for each proportion of the training set                  | `stats/fewshot/<corpus>@<task>.pdf`                                                           |
| [ShowPredictionErrors.py](./ShowPredictionErrors.py)                     | Shows the test examples that a run got wrong, and the classes it confused | printed                                                                                       |
| [ProbeBiasesWithMaskFilling.py](./ProbeBiasesWithMaskFilling.py)         | Lists the words the models predict about groups of patients               | `stats/bias.json`, `stats/bias.tex`                                                           |
| [MeasureVocabularyOverlap.py](./MeasureVocabularyOverlap.py)             | Measures how much the vocabularies of the models overlap                  | `stats/matrix.txt`, `stats/matrix_Blues.png`                                                  |
| [CountSubwordsPerWord.py](./CountSubwordsPerWord.py)                     | Counts the sub-word tokens per word of each dataset, for each model       | `stats/tokens.json`, LaTeX rows printed                                                       |
| [TokenizeBiomedicalTerms.py](./TokenizeBiomedicalTerms.py)               | Shows how the models split some biomedical terms                          | printed                                                                                       |
| [CountClassesOfDatasets.py](./CountClassesOfDatasets.py)                 | Counts and draws the classes of each dataset                              | `stats/classes_distribution.json`, `stats/distributions/<task>.png`                           |

## From the runs to the tables

Each run of a recipe writes its results in `recipes/<recipe>/runs/DrBenchmark-<task name>-<hash>.json`. The tools below turn these files into the scores of the benchmark. `CollectAndAverageScores.py` has to be run first: the other ones read what it writes.

```bash
python tools/CollectAndAverageScores.py    # stats/results.json
python tools/CheckNumberOfRuns.py          # which tasks do not have their 4 runs?
python tools/BuildExcelTableOfScores.py    # stats/overall_results_4runs.xlsx
```

### CollectAndAverageScores.py

Reads every run of every recipe and writes:

- `stats/results.json`: the scores of each run, by model, then by `<corpus>|<task>|<proportion of the training set>`, then by metric;
- `stats/overall_averaged_metrics.json`: their averages, for the runs on the whole training set;
- `stats/overall_std_metrics.json`: their standard deviations (the ones of a sample, n - 1), which are `null` for one run.

The scores that are kept depend on the kind of task:

| Kind of task                             | Scores                              |
|:-----------------------------------------|:------------------------------------|
| POS tagging, named-entity recognition    | `overall_f1`, `overall_accuracy`    |
| Classification (multi-class, multi-label) | `macro_f1`, `weighted_f1`          |
| Multiple-choice question answering       | `hamming_score`, `exact_match`      |
| Semantic textual similarity (regression) | `edrm`, `spearman_correlation_coef` |

Two things are checked on the way, and a third one is reported.

**The runs that are averaged were trained in the same way.** The runs of a task for a model have to share their hyper-parameters. When they do not, nothing is written: the tool lists the hyper-parameters that differ, their values and the runs that have them. Either move the runs that should not be averaged out of the `runs/` directories, or accept a difference with `--ignore`:

```bash
python tools/CollectAndAverageScores.py --ignore max_position_embeddings
```

The seed, the paths and the names of the model and of the task are not compared. Two computed values are: `effective_batch_size` (`batch_size` x `gradient_accumulation_steps`, so that 12 examples per batch and 12 accumulated batches of one example are the same training), and `updates_per_epoch`, the number of updates of the weights per epoch that the run really did, read in the state of its trainer when it has it.

**A model has one name.** In offline mode a run only knows the local copy of its model, `../../../models/dr-bert_drbert-7gb`. The recent runs also have its name on the HuggingFace Hub (`hub_model_name`); for the other ones, it is looked for in `models.txt`, then in the recent runs. When it is not found, the name is rebuilt from the directory, in lower case.

**The versions that computed the runs.** A run writes the versions of Python and of the libraries it was computed with, the commit of the repository, and whether the scripts were changed since that commit. The versions that are not the same for every run are listed, with the number of runs of each one: the runs are averaged all the same. The first runs did not write them: their versions are `unknown`.

### CheckNumberOfRuns.py

Prints the scores of `stats/results.json` that do not come from the expected number of runs: one line per model, task and metric, with the number of runs it has. Nothing is printed when every task has its runs. The number of runs is 4, or `--nb-run`.

### WriteSlurmScriptForMissingRuns.py

Prints a script for SLURM with one job per missing run, for each model of `stats/results.json` and each task of `run.py`, so that all of them have 4 runs (`--nb-run`). It has to be submitted from the root of the repository. Its jobs call the launchers of the recipes, which draw the seed of each run: they add runs to the ones that exist, whatever their seeds. After `run_all_slurm.sh` or `run_all_jean_zay.sh`, whose runs have the seeds of `run.py`, submitting the same script again is enough: it only computes the runs that are missing.

```bash
python tools/WriteSlurmScriptForMissingRuns.py > missing_runs.sh
sbatch missing_runs.sh
```

`--jeanzay` writes the header for Jean Zay, and `--filter` leaves out the jobs whose command contains one of the given texts (a recipe, a model).

### BuildExcelTableOfScores.py

Builds the table of the benchmark: one column per model and one row per score of each task but DEFT-2019, with the average of 4 runs (`--nb-run`). The score is the F1 of the task (`overall_f1` or `weighted_f1`); the question answering and the similarity tasks have two rows, for their two scores. A model that has fewer runs on a task has no score for it; when it has more, 4 of them are drawn, always the same ones. The table is written in `stats/overall_results_<n>runs.xlsx`, or printed in CSV with `--csv`. `--results` gives another file than `stats/results.json`. The standard deviations of the same runs are in the second sheet of the Excel file, `std`, and with `--csv --std` in a column after the average of each model.

### BuildLatexTableOfScores.py

Prints the LaTeX table of the averaged scores (`stats/overall_averaged_metrics.json`, or `--results`): one row per task of the first model, one column per model of `models.txt`, or of `--models`, in that order. With `--std`, each score is followed by the standard deviation of its runs (`85.12 ± 0.34`), read in `stats/overall_std_metrics.json` or in the file given after `--std`.

### PlotFewShotScores.py

Draws, for each task, the score of the models when they are trained on 25%, 50%, 75% and 100% of the training set: `stats/fewshot/<corpus>@<task>.pdf`. The runs on a part of the training set are the ones started with `--fewshot 0.25`, `0.5` or `0.75`; a run with another proportion makes the tool fail. Only the models that have a short name in the `mapping` of the tool are drawn. `--results` gives another file than `stats/results.json`.

### ShowPredictionErrors.py

Shows what a run got wrong on the test set. The results of a run have its predictions, but neither the text of the examples nor, for some tasks, the names of the classes: the test set is read again (from the HuggingFace Hub, or from its local copy with `--offline`).

```bash
python tools/ShowPredictionErrors.py --task cas-pos --model Dr-BERT/DrBERT-7GB
python tools/ShowPredictionErrors.py --run recipes/cas/runs/DrBenchmark-cas-pos-8c05112b3a9ba5af.json
```

The run is the one of a task for a model on the whole training set, with the lowest seed or the one of `--seed`, or the one of a results file (`--run`). The tool prints:

- the confusions, the most frequent ones first (`--max-confusions`, 20 by default): `expected -> predicted: number of times`. For a multi-label task, a class is `(missing)` or added to `(none)`;
- the examples with an error (`--max-examples`, 20 by default), with what was expected and what was predicted. A sentence of a tagging task is shown with its wrong words: `toux [B-NOM -> B-ADJ]`. The other tasks show the texts of the example, cut at 300 characters (`--max-length`).

A similarity score has no class: the examples are the ones whose predicted score is the farthest from the expected one.

## Analyses of the models and of the datasets

These tools do not need any run. They read the models of `models.txt`, or the ones given with `--models` (names on the HuggingFace Hub, or paths), and the datasets of the benchmark but DEFT-2019.

By default the models and the datasets are downloaded from the HuggingFace Hub. With `--offline`, they are read where `download_models_locally.py` and `download_datasets_locally.py` save them (`models/` and `recipes/<recipe>/data/`), as the recipes do in offline mode.

```bash
python tools/CountSubwordsPerWord.py --offline
python tools/MeasureVocabularyOverlap.py --models almanach/camembert-base Dr-BERT/DrBERT-7GB
```

In the tables and the figures, a model has the short name its tool gives it in its `mapping` (`DrBERT-FS` for `Dr-BERT/DrBERT-7GB`, for instance), or its own name when it has none.

### ProbeBiasesWithMaskFilling.py

Asks each model to fill the masked word of sentences about patients of a gender, an origin or a religion ("généralement les patients asiatiques sont <mask>"). For each kind of bias and each model, the 15 words of more than 4 characters that are predicted the most often are written in `stats/bias.json`, and in `stats/bias.tex` as the rows of a LaTeX table.

### MeasureVocabularyOverlap.py

Compares the vocabularies of the tokenizers two by two: the percentage of tokens they share, among the tokens of both, is written in `stats/matrix.txt` (one row and one column per model) and drawn in `stats/matrix_Blues.png`.

### CountSubwordsPerWord.py

Tokenizes the test set of each dataset with the tokenizer of each model, and writes the number of sub-word tokens per word of every example in `stats/tokens.json`. The averages are printed as the rows of a LaTeX table, one per dataset.

### TokenizeBiomedicalTerms.py

Prints how each model splits 15 biomedical terms ("bronchopneumopathie", "IRM"...), then a LaTeX table with one row per term and one column per model: the sub-word tokens of the term, or a check mark when the tokenizer keeps it whole. Without `--models`, the models are the six ones listed at the top of the tool.

### CountClassesOfDatasets.py

Counts the classes of the training, validation and test sets of each task, and writes them in `stats/classes_distribution.json`. For the two similarity tasks, the values to predict are kept instead. One figure per task is written in `stats/distributions/`: the three sets are drawn in turn in the same file, which ends with the distribution of the test set.

## Shared code

These files are not run: the tools import them.

- [repository_paths.py](./repository_paths.py): where the files of the repository are.
- [number_of_runs.py](./number_of_runs.py): the number of runs of each task for each model, 4.
- [models_and_datasets.py](./models_and_datasets.py): the models of `models.txt`, and the datasets, from the HuggingFace Hub or from their local copies.
