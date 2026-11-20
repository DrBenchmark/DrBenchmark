# Tutorial: train a task, collect its scores, test the model on a sentence

This tutorial goes through the benchmark with one task and one model, from the installation to a model that labels a sentence of yours:

1. [train a model on a task](#1-train-a-model-on-a-task): [DrBERT](https://huggingface.co/Dr-BERT/DrBERT-7GB) learns to recognise the entities of the MEDLINE titles of [QUAERO](./recipes/quaero/), which is the task `quaero-ner_medline`;
2. [do a second run](#2-do-a-second-run), as a score of the benchmark is the average of several runs;
3. [collect the scores](#3-collect-the-scores) of the runs;
4. [test the model on a sentence](#4-test-the-model-on-a-sentence) that is typed;
5. [look at the errors of the model](#5-look-at-the-errors-of-the-model) on the test set;
6. [free the disk space](#6-free-the-disk-space) that the models take.

The task is a small one, with 833 titles to train on, and the tutorial trains for 5 epochs: a run takes a little more than one minute on a laptop with a GPU, and more without one, once the model and the dataset are downloaded. The outputs that are shown are the ones of such a laptop, a MacBook Pro (M1 Pro). Yours have other dates and other paths, and scores that are close to these ones: a training does not give exactly the same numbers on every machine.

## Before starting

```bash
git clone https://github.com/DrBenchmark/DrBenchmark
cd DrBenchmark
conda create -n DrBenchmark python pip
conda activate DrBenchmark
pip install -r requirements.txt
```

The commands of the tutorial are typed at the root of the repository. The first run downloads the model and the dataset from the HuggingFace Hub: nothing else has to be prepared. Each run keeps a model of 445 MB.

## 1. Train a model on a task

```bash
python run.py --tasks quaero-ner_medline --models Dr-BERT/DrBERT-7GB --offline false --epochs 5
```

- `--tasks`: the task, one of the 27 names that `python run.py --help` lists and that [recipes/README.md](./recipes/README.md) describes. Several names can be given, or `all`.
- `--models`: the model to fine-tune, by its name on the HuggingFace Hub or by the path of its directory. Without `--models`, the models are the ones of `models.txt`.
- `--offline false`: the dataset and the model are downloaded from the HuggingFace Hub. Without it they are read from local copies, which have to be made first: this is the default mode of the repository, for a machine without network access (see [Running the benchmark](./README.md#running-the-benchmark)).
- `--epochs 5`: 5 epochs instead of the 15 of the task, to make the tutorial short. The hyper-parameters of a task are in its YAML file, [recipes/quaero/yaml/ner.yaml](./recipes/quaero/yaml/ner.yaml): a run of the benchmark is given none on its command line.

```plain
2026-11-20 19:04:27 INFO::=============================
2026-11-20 19:04:27 INFO::Evaluating Dr-BERT/DrBERT-7GB on quaero-ner_medline (run 0, seed 0)
2026-11-20 19:04:27 INFO::=============================
...
{'eval_loss': 0.7525282502174377, 'eval_precision': 0.53500823723229, 'eval_recall': 0.5781041388518025, 'eval_f1': 0.5557219251336898, 'eval_accuracy': 0.8424259735313719, ..., 'epoch': 5.0}
{'train_runtime': 63.5726, ..., 'epoch': 5.0}
...
2026-11-20 19:05:45 INFO::1 run(s): 1 computed, 0 already computed, 0 failed
```

The model is trained on the training set, and evaluated on the validation set after each epoch: the `eval_` scores above are the ones of the last epoch. The checkpoint with the best score on the validation set is the model of the run: it is evaluated on the test set, and kept. The run writes:

- its results, `recipes/quaero/runs/DrBenchmark-quaero-ner_medline-3af4cf7540034bff.json`: its scores and its predictions on the test set, its hyper-parameters and the versions that computed it;
- its model, `recipes/quaero/save_models/DrBenchmark-quaero-ner_medline-3af4cf7540034bff_best_model/`.

The name of a run is built from its arguments: the task, the model, the seed and the hyper-parameters. The same command names the same run, on every machine.

## 2. Do a second run

The same command, a second time, computes nothing: the results of its run exist.

```bash
python run.py --tasks quaero-ner_medline --models Dr-BERT/DrBERT-7GB --offline false --epochs 5
```

```plain
2026-11-20 19:05:45 INFO::=============================
2026-11-20 19:05:45 INFO::Evaluating Dr-BERT/DrBERT-7GB on quaero-ner_medline (run 0, seed 0)
2026-11-20 19:05:45 INFO::=============================
...
This configuration was already computed (seed: 0, model: Dr-BERT/DrBERT-7GB).
...
2026-11-20 19:05:45 INFO::1 run(s): 0 computed, 1 already computed, 0 failed
```

A score of the benchmark is the average of several runs that only differ by their seed: 4 runs in the paper. `run.py` gives the seed 0 to the first run of a task for a model, the seed 1 to the second one, and so on. With `--nb-run 2`, the first run is skipped and the second one is computed:

```bash
python run.py --tasks quaero-ner_medline --models Dr-BERT/DrBERT-7GB --offline false --epochs 5 --nb-run 2
```

```plain
2026-11-20 19:05:45 INFO::Evaluating Dr-BERT/DrBERT-7GB on quaero-ner_medline (run 0, seed 0)
...
This configuration was already computed (seed: 0, model: Dr-BERT/DrBERT-7GB).
...
2026-11-20 19:05:45 INFO::Evaluating Dr-BERT/DrBERT-7GB on quaero-ner_medline (run 1, seed 1)
...
2026-11-20 19:07:01 INFO::2 run(s): 1 computed, 1 already computed, 0 failed
```

A benchmark that was interrupted is resumed in the same way: the same command only computes the runs that are missing. A run with other options, `--epochs 15` for instance, is another run, with its own results and its own model.

## 3. Collect the scores

```bash
python tools/CollectAndAverageScores.py
```

```plain
Reading run results...
Dumping to stats/results.json
Averaging runs...
Dumping to stats/overall_averaged_metrics.json
Dumping to stats/overall_std_metrics.json
```

The tool reads the results of every run of every recipe. `stats/results.json` has the scores of each run on the test set, by model then by task. The two other files have their averages and their standard deviations.

```json
{
    "Dr-BERT/DrBERT-7GB": {
        "quaero|ner_medline|1.0": {
            "overall_f1": [
                0.5588420390182506,
                0.5599662162162163
            ],
            "overall_accuracy": [
                0.8404010670591482,
                0.8379173949038727
            ]
        }
    }
}
```

`1.0` is the part of the training set that the runs were trained on: the whole of it.

The table of the benchmark has one row for each score of each task and one column for each model, with the average of its runs:

```bash
python tools/BuildExcelTableOfScores.py --nb-run 2 --csv --std
```

```plain
dataset,task,metric,Dr-BERT/DrBERT-7GB,Dr-BERT/DrBERT-7GB (std)
CAS,POS,F1,,
ESSAI,POS,F1,,
QUAERO,NER EMEA,F1,,
QUAERO,NER MEDLINE,F1,55.94,0.08
E3C,NER Clinical,F1,,
...
```

The F1 of the two runs is 55.94 on average, and their standard deviation is 0.08. The other tasks have no run. The table expects 4 runs for a task: `--nb-run 2` is for the two runs of the tutorial. Without `--csv`, it is written in an Excel file, `stats/overall_results_2runs.xlsx`. The other tools that read the runs are described in [tools/README.md](./tools/README.md).

## 4. Test the model on a sentence

Each run kept its best model. `predict.py` lists the runs and their models:

```bash
python predict.py --list
```

```plain
task                model               seed  validation         test                        kept model
quaero-ner_medline  Dr-BERT/DrBERT-7GB  0     0.8424 (accuracy)  f1 0.5588, accuracy 0.8404  yes, taken
quaero-ner_medline  Dr-BERT/DrBERT-7GB  1     0.8400 (accuracy)  f1 0.5600, accuracy 0.8379  yes

validation: score, on the validation set, of the checkpoint that the run kept. For a task and a model, the model that is taken is the one of the run with the best one, among the runs on the whole training set.
```

The model that predicts is the one of the run of seed 0, which has the best score on the validation set. `--seed 1` takes the other one.

```bash
python predict.py --task quaero-ner_medline --model Dr-BERT/DrBERT-7GB "Une femme de 44 ans était hospitalisée pour une toux sèche."
```

```plain
Model: Dr-BERT/DrBERT-7GB fine-tuned on quaero-ner_medline, seed 0 (validation accuracy: 0.8424, the best of the 2 runs that kept their model)
Une [femme de 44 ans]/LIVB était hospitalisée pour une [toux sèche]/DISO .
```

The words between brackets are an entity, followed by its class: a living being (`LIVB`) and a disorder (`DISO`), two of the ten classes of QUAERO, which [its README](./recipes/quaero/README.md) lists. The other words are in no entity.

- Without a sentence, `predict.py` asks for sentences, until an empty line.
- `--input sentences.txt` labels the sentences of a file, and `--evaluate corpus.conll` scores the model on an annotated corpus of yours: see [Using a trained model](./README.md#using-a-trained-model).
- The notebook [predict_demo.ipynb](./predict_demo.ipynb) highlights the entities of a sentence that is typed.

## 5. Look at the errors of the model

```bash
python tools/ShowPredictionErrors.py --task quaero-ner_medline --model Dr-BERT/DrBERT-7GB --max-confusions 5 --max-examples 2
```

```plain
Run: /path/to/DrBenchmark/recipes/quaero/runs/DrBenchmark-quaero-ner_medline-3af4cf7540034bff.json
Task quaero-ner_medline, model Dr-BERT/DrBERT-7GB, seed 0 (other seeds: 1)
833 examples of the test set, 632 with an error

Confusions: 1735 wrong labels among the ones of 10871 words
    expected -> predicted: number of times
    I-DISO -> O: 138
    B-PROC -> O: 86
    O -> B-PROC: 86
    I-PROC -> O: 75
    O -> I-DISO: 66
    ... and 171 other confusions, 1284 times in all

Examples with an error: 2 of 632

[0] 3 wrong labels
    Etude du contenu du trou [O -> B-ANAT] transversaire [O -> I-ANAT] chez l' Africain [B-LIVB -> B-GEOG] à Dakar

[1] 1 wrong label
    Les gangrènes gazeuses [I-DISO -> B-CHEM]
```

The results of a run have its predictions on the test set. The tool takes the run with the lowest seed (`--seed` takes another one), reads the test set again, and shows the labels that the run confused, then the sentences where it was wrong: a wrong word is followed by the label that was expected and by the one that was predicted. `B-` is the label of the first word of an entity, `I-` the one of its next words, and `O` the one of a word that is in no entity.

## 6. Free the disk space

The two runs kept 890 MB of models, and `predict.py` takes one of them. `tools/RemoveKeptModels.py` lists the other ones:

```bash
python tools/RemoveKeptModels.py
```

```plain
/path/to/DrBenchmark/recipes/quaero/save_models/DrBenchmark-quaero-ner_medline-e5de16c4954b4c11_best_model  445.3 MB  (quaero-ner_medline, Dr-BERT/DrBERT-7GB, seed 1)
1 model to remove, 445.3 MB: 1 model stays. Nothing was removed: give --remove to remove them.
```

`--remove` removes them. The results of the runs stay: their scores are collected as before. With `--keep_model false`, `run.py` does not keep the model of a run at all.

## What next

**The runs of the tutorial are not runs of the benchmark**: they were trained for 5 epochs. The scores of runs that were not trained in the same way are not averaged: `CollectAndAverageScores.py` stops, and lists what differs. Before the benchmark is run for this task and this model, remove the two runs of the tutorial: the models that the runs of the task kept for the model, then their results.

```bash
python tools/RemoveKeptModels.py --task quaero-ner_medline --model Dr-BERT/DrBERT-7GB --all --remove
rm recipes/quaero/runs/DrBenchmark-quaero-ner_medline-3af4cf7540034bff.json recipes/quaero/runs/DrBenchmark-quaero-ner_medline-e5de16c4954b4c11.json
```

- **The whole benchmark**: `python run.py --tasks all --nb-run 4` does 4 runs of every task for each model of `models.txt`, which is where [other models](./README.md#evaluate-other-models) are added. On a cluster, the scripts for SLURM do one run per job: see [Running with SLURM](./README.md#running-with-slurm).
- **Without network access**: `python download_datasets_locally.py` and `python download_models_locally.py` save the datasets and the models where the runs read them in offline mode, which is the default one. `run.py` is then typed without `--offline false`, and `ShowPredictionErrors.py` with `--offline`.
- **Your own data**: `predict.py` labels the sentences of a file, and scores a model on an annotated corpus: see [Using a trained model](./README.md#using-a-trained-model).
- **A task of your own**: [recipes/ADDING_A_TASK.md](./recipes/ADDING_A_TASK.md) adds a task to the benchmark, from a template recipe.
