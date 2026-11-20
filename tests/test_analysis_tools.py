"""Tests of the tools that analyse the models, the datasets and the scores (`tools/`).

Each tool is run once as a script, from a directory which is not the repository.
Nothing is downloaded: the models are tiny ones, the datasets are the synthetic ones,
read in offline mode, and the scores are written by hand. The tests only check that
the tools run and that they write what they are meant to.
"""

import json
import os
import re
import shutil
from collections import Counter

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_from_disk

import synthetic
from helpers import REPO, Task, copy_tools, load_module, read_json, run_tool

N_TEST = synthetic.SPLIT_SIZES["test"]
# The datasets of the benchmark, and their tasks, but the ones of DEFT-2019
N_DATASETS, N_TASKS = 24, 26


@pytest.fixture
def repository(tmp_path):
    """A copy of the tools in a directory that stands for the repository: they read and write there."""
    repository = tmp_path / "repository"
    repository.mkdir()
    copy_tools(repository)
    return repository


@pytest.fixture
def elsewhere(tmp_path):
    """The directory the tools are run from."""
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    return directory


@pytest.fixture
def offline_datasets(repository, synthetic_data):
    """The synthetic datasets, where the tools read the datasets in offline mode.

    `download_datasets_locally.py` saves them in `recipes/<recipe>/data/local_hf_<subset>/`.
    """
    for recipe in synthetic_data.iterdir():
        (repository / "recipes" / recipe.name).mkdir(parents=True)
        os.symlink(recipe, repository / "recipes" / recipe.name / "data")


@pytest.fixture(scope="module")
def tiny_masked_lm(tmp_path_factory):
    """A tiny BERT that predicts masked words.

    `ProbeBiasesWithMaskFilling.py` asks for the 500 most probable words: the vocabulary has more.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("models") / "tiny-masked-lm"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"word{i:03d}" for i in range(600)]
    vocab = {token: index for index, token in enumerate(tokens)}
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]", model_max_length=512,
    ).save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=512,
    )
    BertForMaskedLM(config).save_pretrained(path)
    return path


@pytest.mark.e2e
def test_probe_biases_with_mask_filling(repository, elsewhere, tiny_masked_lm):
    model = str(tiny_masked_lm)
    completed = run_tool("ProbeBiasesWithMaskFilling.py", repository, "--models", model, cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    bias = read_json(repository / "stats" / "bias.json")
    assert set(bias) == {"gender", "race", "religion"}
    for words_of_models in bias.values():
        assert list(words_of_models) == [model]
        # The 15 words of more than 4 characters that are predicted the most often
        assert len(words_of_models[model]) == 15
        assert all(len(word) > 4 for word in words_of_models[model])

    # One row of the LaTeX table per kind of bias and per model
    rows = (repository / "stats" / "bias.tex").read_text()
    assert rows.count("\\multirow{1}{*}{") == 3
    assert rows.count(f"& {model} & ") == 3
    assert not list(elsewhere.iterdir())


@pytest.mark.e2e
def test_measure_vocabulary_overlap(repository, elsewhere, tiny_bert, tiny_flaubert):
    completed = run_tool(
        "MeasureVocabularyOverlap.py", repository, "--models", str(tiny_bert), str(tiny_flaubert), cwd=elsewhere,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    # One row and one column per model: the percentage of tokens that two vocabularies share
    matrix = [
        [float(value) for value in row.split("\t")]
        for row in (repository / "stats" / "matrix.txt").read_text().splitlines()
    ]
    assert [len(row) for row in matrix] == [2, 2]
    assert matrix[0][0] == matrix[1][1] == 100.0
    assert matrix[0][1] == matrix[1][0]
    assert 0 <= matrix[0][1] < 100
    assert (repository / "stats" / "matrix_Blues.png").stat().st_size > 0
    assert not list(elsewhere.iterdir())


@pytest.mark.e2e
def test_models_of_the_models_file_are_read_offline(repository, elsewhere, tiny_bert):
    """Without `--models`, the models are the ones of `models.txt`; with `--offline`, their local copies are read."""
    # Where `download_models_locally.py` saves the model
    shutil.copytree(tiny_bert, repository / "models" / "some-org_some-model")
    # Neither the empty lines nor the comments are models
    (repository / "models.txt").write_text("# The models to evaluate\n\n   # not/this-one\n  Some-Org/Some-Model  \n#nor/this-one")

    completed = run_tool("MeasureVocabularyOverlap.py", repository, "--offline", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines()[0] == "Some-Org/Some-Model - Some-Org/Some-Model : 100.0"
    assert (repository / "stats" / "matrix.txt").read_text() == "100.0\n"


@pytest.mark.e2e
def test_missing_model_in_offline_mode(repository, elsewhere):
    """A tool says which script saves the copy of a model that it does not find, instead of failing on a directory."""
    completed = run_tool("MeasureVocabularyOverlap.py", repository, "--offline", "--models", "Some-Org/Some-Model", cwd=elsewhere)
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert f"its copy is not in {repository / 'models' / 'some-org_some-model'}, where it is read with --offline." in completed.stderr
    assert f"Save it with `python {repository / 'download_models_locally.py'} --models Some-Org/Some-Model`" in completed.stderr


@pytest.mark.e2e
def test_missing_dataset_in_offline_mode(repository, elsewhere):
    """A tool says which script saves the copy of a dataset that it does not find."""
    completed = run_tool("CountClassesOfDatasets.py", repository, "--offline", cwd=elsewhere)
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    data = repository / "recipes" / "quaero" / "data" / "local_hf_emea"
    assert f"The dataset DrBenchmark/QUAERO (subset emea) is not in {data}, where it is read with --offline." in completed.stderr
    assert f"`python {repository / 'download_datasets_locally.py'}`" in completed.stderr
    assert not (repository / "stats" / "classes_distribution.json").exists()


@pytest.mark.e2e
def test_count_subwords_per_word(repository, elsewhere, offline_datasets, tiny_bert):
    model = str(tiny_bert)
    completed = run_tool("CountSubwordsPerWord.py", repository, "--models", model, "--offline", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    # Number of tokens per word, for each example of the test set of each dataset
    tokens = read_json(repository / "stats" / "tokens.json")
    assert len(tokens) == N_DATASETS
    for dataset, tokens_of_models in tokens.items():
        assert dataset.startswith("DrBenchmark/")
        assert list(tokens_of_models) == [model]
        assert len(tokens_of_models[model]) == N_TEST, dataset
        # At least one token per word, and the special tokens
        assert all(tokens_per_word > 1 for tokens_per_word in tokens_of_models[model]), dataset

    # The LaTeX table: the names of the models, then one row per dataset
    table = completed.stdout[completed.stdout.index("JSON file saved!"):].splitlines()[1:]
    assert table[0] == f" & {model}\\\\"
    assert len(table) == 1 + N_DATASETS
    assert all(re.fullmatch(r"[\w ]+ & \d+\.\d+ \\\\", row) for row in table[1:]), table
    assert not list(elsewhere.iterdir())


@pytest.mark.e2e
def test_count_classes_of_datasets(repository, elsewhere, offline_datasets):
    completed = run_tool("CountClassesOfDatasets.py", repository, "--offline", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    distribution = read_json(repository / "stats" / "classes_distribution.json")
    assert len(distribution) == N_TASKS
    for task, splits in distribution.items():
        assert task.startswith("DrBenchmark/")
        assert set(splits) == {"train", "validation", "test"}
        # The classes of each set, with their number of occurrences, or the values to predict
        assert all(splits.values()), task

    # One figure per task
    figures = list((repository / "stats" / "distributions").glob("*.png"))
    assert len(figures) == N_TASKS
    assert all(figure.stat().st_size > 0 for figure in figures)
    assert not list(elsewhere.iterdir())


@pytest.mark.e2e
def test_tokenize_biomedical_terms(repository, elsewhere, tiny_bert, tiny_flaubert):
    models = [str(tiny_bert), str(tiny_flaubert)]
    completed = run_tool("TokenizeBiomedicalTerms.py", repository, "--models", *models, cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    table = completed.stdout[completed.stdout.index("\\begin{table}"):]
    # One column for the terms and one per model, named in the header
    assert "\\begin{tabular}{ccc}\n" in table
    assert f"\\textbf{{Term}} &\n\\textbf{{{models[0]}}} &\n\\textbf{{{models[1]}}} \\\\\n" in table
    # One row per term: its tokens for each model, or a check mark when it is not split
    rows = re.findall(r"^\\textit\{(\w+)\} & (\S+) & (\S+) \\\\ $", table, re.M)
    assert len(rows) == 15
    assert rows[0][0] == "asymptomatique"
    assert all("-" in tokens or tokens == "\\checkmark" for _, *row in rows for tokens in row)


def test_build_latex_table_of_scores(repository, elsewhere):
    (repository / "stats" / "overall_averaged_metrics.json").write_text(json.dumps({
        "Dr-BERT/DrBERT-7GB": {
            "cas|pos|1.0": {"overall_f1": 0.9, "overall_accuracy": 0.95},
            "cas|cls|1.0": {"macro_f1": 0.5, "weighted_f1": 0.625},
            "clister|regr|1.0": {"edrm": 0.75, "spearman_correlation_coef": 0.8},
            "frenchmedmcqa|mcqa|1.0": {"hamming_score": 0.3, "exact_match": 0.125},
        },
        "another/model": {"cas|pos|1.0": {"overall_f1": 0.8, "overall_accuracy": 0.85}},
    }))
    completed = run_tool(
        "BuildLatexTableOfScores.py", repository, "--models", "Dr-BERT/DrBERT-7GB", "another/model", cwd=elsewhere,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    table = completed.stdout[completed.stdout.index("\\begin{table*}"):]
    # Two columns for the dataset and the task, and one per model: a model without a short name keeps its own
    assert "\\begin{tabular}{|l|l|c|c|}\n" in table
    assert "\\textbf{Dataset} & \\textbf{Task} & \\textbf{DrBERT-FS} & \\textbf{another/model} \\\\ " in table
    # One row per task of the first model, with the score of its kind of task
    assert "\\multirow{2}{*}{CAS} & POS & 90.0 & 80.0 \\\\ " in table
    assert " & CLS & 62.5 & - \\\\ " in table
    assert "\\multirow{2}{*}{CLISTER} & REGR & 0.75 / 0.8 & - \\\\ " in table
    assert "\\multirow{2}{*}{FRENCHMEDMCQA} & MCQA & 30.0 / 12.5 & - \\\\ " in table
    assert table.count(" \\\\ ") == 5


def test_build_latex_table_of_the_models_file(repository, elsewhere):
    """Without `--models`, the columns are the models of `models.txt`, but its comments."""
    (repository / "models.txt").write_text("# The models to evaluate\nalmanach/camembert-base\n\nDr-BERT/DrBERT-7GB\n")
    scores = {"cas|pos|1.0": {"overall_f1": 0.5, "overall_accuracy": 0.5}}
    (elsewhere / "averages.json").write_text(json.dumps({"almanach/camembert-base": scores, "Dr-BERT/DrBERT-7GB": scores}))

    # A path given on the command line is relative to the current directory
    completed = run_tool("BuildLatexTableOfScores.py", repository, "--results", "averages.json", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "\\textbf{Dataset} & \\textbf{Task} & \\textbf{CamemBERT} & \\textbf{DrBERT-FS} \\\\ " in completed.stdout
    assert "\\begin{tabular}{|l|l|c|c|}\n" in completed.stdout


def test_plot_few_shot_scores(repository, elsewhere):
    def scores(base):
        """Scores of two runs of two tasks, for each proportion of the training set."""
        metrics_of_tasks = {"cas|pos": ("overall_f1", "overall_accuracy"), "clister|regr": ("edrm", "spearman_correlation_coef")}
        return {
            f"{task}|{proportion}": {metric: [base + proportion / 4, base + proportion / 2] for metric in metrics}
            for task, metrics in metrics_of_tasks.items()
            for proportion in (0.25, 0.5, 0.75, 1.0)
        }

    (repository / "stats" / "results.json").write_text(json.dumps({
        "Dr-BERT/DrBERT-7GB": scores(0.3),
        "almanach/camembert-base": scores(0.2),
        # A model without a name for the figures is not drawn
        "another/model": scores(0.1),
    }))
    completed = run_tool("PlotFewShotScores.py", repository, cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    # One figure per task
    figures = repository / "stats" / "fewshot"
    assert sorted(path.name for path in figures.iterdir()) == ["cas@pos.pdf", "clister@regr.pdf"]
    assert all(path.read_bytes().startswith(b"%PDF") for path in figures.iterdir())
    assert completed.stdout.splitlines() == [str(figures / "cas@pos.pdf"), str(figures / "clister@regr.pdf")]
    assert not list(elsewhere.iterdir())


def test_kinds_of_predictions_and_names_of_their_classes(monkeypatch):
    """The predictions of a run say which kind they are, and the test set gives the names of the classes written as numbers."""
    # The tools import the modules they share from their own directory
    monkeypatch.syspath_prepend(str(REPO / "tools"))
    tool = load_module("show_prediction_errors", REPO / "tools" / "ShowPredictionErrors.py")

    assert tool.kind_of([["B-age", "O"], ["O"]]) == tool.TOKENS
    assert tool.kind_of([[0.0, 1.0], [1.0, 1.0]]) == tool.LABEL_SETS
    assert tool.kind_of([2, 0]) == tool.CLASSES
    assert tool.kind_of(["ab", "c"]) == tool.ANSWERS
    assert tool.kind_of([3.5, 1.0]) == tool.SCORES

    test = Dataset.from_dict(
        {"id": ["a", "b", "c"], "source": [0, 1, 0], "label": [2, 0, 1], "specialities": [[0], [1, 2], []], "text": ["x", "y", "z"]},
        features=Features({
            "id": Value("string"), "source": ClassLabel(names=["journal", "report"]),
            "label": ClassLabel(names=["first", "second", "third"]),
            "specialities": Sequence(ClassLabel(names=["cardiology", "urology", "oncology", "pneumology"])), "text": Value("string"),
        }),
    )
    # The classes of a run are the ones of the column that has its references: not the first column of classes
    assert tool.names_of_classes(test, tool.CLASSES, [2, 0, 1]) == ["first", "second", "third"]
    assert tool.names_of_classes(test, tool.CLASSES, [0, 1, 0]) == ["journal", "report"]
    # With `--max_test_samples`, a run only has the first examples
    assert tool.names_of_classes(test, tool.CLASSES, [2, 0]) == ["first", "second", "third"]
    # No column of the test set has these references: the classes keep their numbers
    assert tool.names_of_classes(test, tool.CLASSES, [1, 1, 1]) is None
    # A multi-label task: 0 or 1 for each of the 4 classes of `specialities`
    classes = ["cardiology", "urology", "oncology", "pneumology"]
    assert tool.names_of_classes(test, tool.LABEL_SETS, [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]) == classes
    assert tool.names_of_classes(test, tool.LABEL_SETS, [[1, 0], [0, 1], [0, 0]]) is None


def show_prediction_errors(repository, elsewhere, *arguments):
    return run_tool("ShowPredictionErrors.py", repository, *arguments, "--offline", cwd=elsewhere)


# One task for each kind of predictions that the recipes write: a label per word, a class, classes, a score, letters
@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "quaero-ner_emea", "cas-cls", "morfitt-cls", "clister-regr", "frenchmedmcqa-mcqa"])
def test_show_prediction_errors(task_name, repository, elsewhere, offline_datasets, synthetic_data, run_recipe, tiny_bert):
    """The predictions of a real run are shown with the texts and the names of the classes of the test set."""
    task = Task(task_name)
    run = run_recipe(task_name, tiny_bert)
    predictions = run.result["predictions"]
    references, predicted = predictions["real_labels"], predictions["system_predictions"]
    test = load_from_disk(str(synthetic_data / task.recipe / f"local_hf_{task.subset}"))["test"]
    assert len(references) == len(test) > 0

    completed = show_prediction_errors(
        repository, elsewhere, "--run", str(run.run_files[0]), "--max-examples", "0", "--max-confusions", "0",
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    assert lines[:2] == [f"Run: {run.run_files[0]}", f"Task {task_name}, model {tiny_bert}, seed 0"]
    # One line per example that is shown: [identifier] ...
    shown = [line for line in lines if line.startswith("[")]

    if task.kind == "regr":
        distances = [abs(reference - prediction) for reference, prediction in zip(references, predicted)]
        assert lines[2] == (
            f"{len(test)} examples of the test set: the predicted score is at {sum(distances) / len(distances):.2f} "
            f"of the expected one on average, and at {max(distances):.2f} at most"
        )
        assert f"Examples with the largest error: {len(test)} of {len(test)}" in lines
        # The farthest from the expected score first
        farthest = max(range(len(test)), key=lambda index: distances[index])
        assert shown[0] == (
            f"[{test[farthest]['id']}] expected: {references[farthest]:.2f} | predicted: {predicted[farthest]:.2f} "
            f"(distance: {distances[farthest]:.2f})"
        )
        assert [float(line.rsplit(" ", 1)[1].rstrip(")")) for line in shown] == sorted((round(d, 2) for d in distances), reverse=True)
        # The two sentences that are compared
        assert f"    text_1: {test[farthest]['text_1']}" in lines and f"    text_2: {test[farthest]['text_2']}" in lines
        return

    if task.kind in ("pos", "ner"):
        confusions = Counter(
            pair for reference, prediction in zip(references, predicted) for pair in zip(reference, prediction) if pair[0] != pair[1]
        )
        described = [
            (f"{sum(a != b for a, b in zip(reference, prediction))} wrong label",
             "    " + " ".join(word if a == b else f"{word} [{a} -> {b}]" for word, a, b in zip(example["tokens"], reference, prediction)))
            for example, reference, prediction in zip(test, references, predicted)
        ]
        assert f"Confusions: {sum(confusions.values())} wrong labels among the ones of {sum(map(len, references))} words" in lines
    elif isinstance(references[0], list):
        # Multi-label classification: 0 or 1 for each class of the column `specialities`
        names = test.features["specialities"].feature.names

        def classes(vector):
            return [name for name, value in zip(names, vector) if value >= 0.5]

        confusions = Counter()
        for reference, prediction in zip(references, predicted):
            confusions.update((name, "(missing)") for name in classes(reference) if name not in classes(prediction))
            confusions.update(("(none)", name) for name in classes(prediction) if name not in classes(reference))
        described = [
            (f"expected: {', '.join(classes(reference)) or '(none)'} | predicted: {', '.join(classes(prediction)) or '(none)'}", None)
            for reference, prediction in zip(references, predicted)
        ]
    else:
        # A class, written as a number by the classification tasks and as letters by the question answering one
        names = test.features["label"].names if task.kind == "cls" else None
        named = [(names[r], names[p]) if names else (r, p) for r, p in zip(references, predicted)]
        confusions = Counter(pair for pair in named if pair[0] != pair[1])
        described = [(f"expected: {expected} | predicted: {prediction}", None) for expected, prediction in named]
    if task.kind != "pos" and task.kind != "ner":
        assert f"Confusions: {sum(confusions.values())}" in lines

    wrong = [index for index, (reference, prediction) in enumerate(zip(references, predicted)) if reference != prediction]
    # A tiny model that was trained for one epoch is wrong
    assert wrong
    assert lines[2] == f"{len(test)} examples of the test set, {len(wrong)} with an error"
    assert f"Examples with an error: {len(wrong)} of {len(wrong)}" in lines
    assert sorted(line for line in lines if " -> " in line and line.startswith("    ") and line.rsplit(": ", 1)[-1].isdigit()) == sorted(
        f"    {expected} -> {prediction}: {count}" for (expected, prediction), count in confusions.items()
    )
    for index, line in zip(wrong, shown, strict=True):
        description, sentence = described[index]
        assert line.startswith(f"[{test[index]['id']}] {description}")
        if sentence is not None:
            assert sentence in lines
    if task_name == "cas-cls":
        # The words of the sentence, which the recipe joins to build the text it classifies
        assert f"    tokens: {' '.join(test[wrong[0]]['tokens'])}" in lines


@pytest.mark.e2e
def test_show_prediction_errors_of_a_task_for_a_model(repository, elsewhere, offline_datasets, run_recipe, tiny_bert):
    """The run is looked for in the results of the recipe: the one of the task, of the model and of the seed."""
    results = run_recipe("cas-cls", tiny_bert).result
    directory = repository / "recipes" / "cas" / "runs"
    directory.mkdir()
    other_runs = (
        (0, {}), (3, {}),
        # A run in offline mode: the copy of the model, and its name on the HuggingFace Hub
        (5, {"model_name": "../../../models/other-org_other-model", "hub_model_name": "Other-Org/Other-Model"}),
        # A run of the first scripts, which did not write the name of the model
        (6, {"model_name": "../../../models/old-org_old-model"}),
        # Runs that are not the ones of the task on the whole training set
        (7, {"fewshot": 0.5}), (8, {"task": "cas-pos"}),
    )
    for seed, changes in other_runs:
        run = json.loads(json.dumps(results))
        run["hyperparameters"].update(seed=seed, **changes)
        (directory / f"DrBenchmark-run-{seed}.json").write_text(json.dumps(run))

    def shown(*arguments):
        completed = show_prediction_errors(repository, elsewhere, "--task", "cas-cls", *arguments)
        assert completed.returncode == 0, completed.stderr[-3000:]
        return completed.stdout.splitlines()

    lines = shown("--model", str(tiny_bert))
    assert lines[:2] == [f"Run: {directory / 'DrBenchmark-run-0.json'}", f"Task cas-cls, model {tiny_bert}, seed 0 (other seeds: 3)"]
    assert shown("--model", str(tiny_bert), "--seed", "3")[1] == f"Task cas-cls, model {tiny_bert}, seed 3 (other seeds: 0)"
    assert shown("--model", "Other-Org/Other-Model")[1] == "Task cas-cls, model Other-Org/Other-Model, seed 5"
    assert shown("--model", "Old-Org/Old-Model")[1] == "Task cas-cls, model ../../../models/old-org_old-model, seed 6"

    # At most two confusions and one example
    lines = shown("--model", str(tiny_bert), "--max-confusions", "1", "--max-examples", "1")
    assert sum(line.startswith("[") for line in lines) == 1
    assert sum(line.startswith("    ") and " -> " in line for line in lines) in (1, 2)

    def refused(*arguments):
        completed = show_prediction_errors(repository, elsewhere, *arguments)
        assert completed.returncode != 0 and "Traceback" not in completed.stderr
        return completed.stderr

    assert "has the seed 9: the seeds of its runs are 0, 3." in refused("--task", "cas-cls", "--model", str(tiny_bert), "--seed", "9")
    unknown = refused("--task", "cas-cls", "--model", "unknown/model")
    assert f"No run of cas-cls for unknown/model in {directory}." in unknown
    # The models that have runs of the task are listed
    models = unknown.strip().splitlines()[-1].split(": ")[1].split(", ")
    assert sorted(models) == sorted(["../../../models/old-org_old-model", "Other-Org/Other-Model", str(tiny_bert)])
    assert "give the run either by --task and --model, or by --run" in refused("--task", "cas-cls")
    assert "give the run either by --task and --model, or by --run" in refused()
