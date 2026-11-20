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

import pytest

import synthetic
from helpers import copy_tools, read_json, run_tool

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

    `BiasMaskFilling.py` asks for the 500 most probable words: the vocabulary has more.
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
def test_bias_mask_filling(repository, elsewhere, tiny_masked_lm):
    model = str(tiny_masked_lm)
    completed = run_tool("BiasMaskFilling.py", repository, "--models", model, cwd=elsewhere)
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
def test_inter_vocabularies_coverage_matrix(repository, elsewhere, tiny_bert, tiny_flaubert):
    completed = run_tool(
        "InterVocabulariesCoverageMatrix.py", repository, "--models", str(tiny_bert), str(tiny_flaubert), cwd=elsewhere,
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
    (repository / "models.txt").write_text("# The models to evaluate\nSome-Org/Some-Model\n")

    completed = run_tool("InterVocabulariesCoverageMatrix.py", repository, "--offline", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines()[0] == "Some-Org/Some-Model - Some-Org/Some-Model : 100.0"
    assert (repository / "stats" / "matrix.txt").read_text() == "100.0\n"


@pytest.mark.e2e
def test_subword_analysis(repository, elsewhere, offline_datasets, tiny_bert):
    model = str(tiny_bert)
    completed = run_tool("SubwordAnalysis.py", repository, "--models", model, "--offline", cwd=elsewhere)
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
def test_classes_distribution(repository, elsewhere, offline_datasets):
    completed = run_tool("ClassesDistribution.py", repository, "--offline", cwd=elsewhere)
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
def test_make_table_tokenization(repository, elsewhere, tiny_bert, tiny_flaubert):
    models = [str(tiny_bert), str(tiny_flaubert)]
    completed = run_tool("MakeTableTokenization.py", repository, "--models", *models, cwd=elsewhere)
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


def test_generate_latex_table(repository, elsewhere):
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
        "GenerateLatexTableFromResultsJSON.py", repository, "--models", "Dr-BERT/DrBERT-7GB", "another/model", cwd=elsewhere,
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


def test_generate_latex_table_of_the_models_file(repository, elsewhere):
    """Without `--models`, the columns are the models of `models.txt`, but its comments."""
    (repository / "models.txt").write_text("# The models to evaluate\nalmanach/camembert-base\n\nDr-BERT/DrBERT-7GB\n")
    scores = {"cas|pos|1.0": {"overall_f1": 0.5, "overall_accuracy": 0.5}}
    (elsewhere / "averages.json").write_text(json.dumps({"almanach/camembert-base": scores, "Dr-BERT/DrBERT-7GB": scores}))

    # A path given on the command line is relative to the current directory
    completed = run_tool("GenerateLatexTableFromResultsJSON.py", repository, "--results", "averages.json", cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "\\textbf{Dataset} & \\textbf{Task} & \\textbf{CamemBERT} & \\textbf{DrBERT-FS} \\\\ " in completed.stdout
    assert "\\begin{tabular}{|l|l|c|c|}\n" in completed.stdout


def test_draw_lines_plots_few_shot(repository, elsewhere):
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
    completed = run_tool("DrawLinesPlotsFewShot.py", repository, cwd=elsewhere)
    assert completed.returncode == 0, completed.stderr[-3000:]

    # One figure per task
    figures = repository / "stats" / "fewshot"
    assert sorted(path.name for path in figures.iterdir()) == ["cas@pos.pdf", "clister@regr.pdf"]
    assert all(path.read_bytes().startswith(b"%PDF") for path in figures.iterdir())
    assert completed.stdout.splitlines() == [str(figures / "cas@pos.pdf"), str(figures / "clister@regr.pdf")]
    assert not list(elsewhere.iterdir())
