"""Tests of `predict.py`, which predicts with the models that the runs kept.

The first tests do not load any model: they read results that are written by hand. The
ones marked `e2e` use the models that the recipes keep after a training on synthetic data:
`predict.py` has to give, for the examples of the test set, the predictions that the run wrote.
"""

import csv
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import types

import pytest

import synthetic
from helpers import ALL_TASKS, RECIPES, REPO, Runs, Task, copy_tools, load_module, read_json, run_command, run_tool

MODEL = "Some-Org/Some-Model"


@pytest.fixture(scope="session")
def predict():
    return load_module("drbenchmark_predict", REPO / "predict.py")


def task_ids(task):
    return task.name


# ---------------------------------------------------------------------------------------------
# What a person types, and what is shown
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, words", [
    ("Le patient présente une toux sèche.", ["Le", "patient", "présente", "une", "toux", "sèche", "."]),
    # The article is a word with its apostrophe, as in CAS, ESSAI, DEFT-2021 and QUAERO
    ("l'examen d’une patiente", ["l'", "examen", "d’", "une", "patiente"]),
    ("selles glairo-sanglantes, anti-PD-1", ["selles", "glairo-sanglantes", ",", "anti-PD-1"]),
    ("imovane 7,5 mg puis 10.25 g/jour", ["imovane", "7,5", "mg", "puis", "10.25", "g", "/", "jour"]),
    ("débuter aujourd'hui (matin)...", ["débuter", "aujourd'hui", "(", "matin", ")", "..."]),
    ("  deux   espaces ", ["deux", "espaces"]),
    ("", []),
])
def test_sentence_is_split_into_words(text, words, predict):
    assert predict.split_words(text) == words


@pytest.mark.parametrize("labels, expected", [
    (["O", "B-DISO", "I-DISO", "O"], [(1, 3, "DISO")]),
    # Two entities of the same type that follow each other
    (["B-DISO", "B-DISO", "I-DISO"], [(0, 1, "DISO"), (1, 3, "DISO")]),
    (["B-DISO", "I-CHEM"], [(0, 1, "DISO"), (1, 2, "CHEM")]),
    # An entity that does not start with B-
    (["O", "I-DISO", "I-DISO"], [(1, 3, "DISO")]),
    # The marks of CAS and ESSAI
    (["B_scope_neg", "I_scope_neg", "O"], [(0, 2, "scope_neg")]),
    # The tags of the parts of speech: each word has its own
    (["B-DET:ART", "B-NOM", "B-NOM"], [(0, 1, "DET:ART"), (1, 2, "NOM"), (2, 3, "NOM")]),
    # Labels without mark: the same one on words that follow each other is one entity
    (["drug", "drug", "O", "dose"], [(0, 2, "drug"), (3, 4, "dose")]),
    ([], []),
])
def test_entities_of_a_sentence(labels, expected, predict):
    assert predict.spans(labels) == expected


def test_sentence_is_shown_with_its_entities(predict):
    words = ["une", "toux", "sèche", "sous", "aspirine", "."]
    labels = ["O", "B-DISO", "I-DISO", "O", "B-CHEM", "O"]
    assert predict.words_with_labels(words, labels) == "une [toux sèche]/DISO sous aspirine/CHEM ."
    assert predict.words_with_labels(["Le", "patient"], ["B-DET:ART", "B-NOM"]) == "Le/DET:ART patient/NOM"
    assert predict.words_with_labels(["rien"], ["O"]) == "rien"


def test_entities_are_highlighted_for_a_notebook(predict):
    html = predict.highlight(["une", "toux", "sèche", "<", "aspirine"], ["O", "B-DISO", "I-DISO", "O", "B-CHEM"])
    marks = re.findall(r"<mark style=\"background: (#\w+);[^>]*>(.*?) <small[^>]*>(.*?)</small></mark>", html)
    assert [(text, kind) for _, text, kind in marks] == [("toux sèche", "DISO"), ("aspirine", "CHEM")]
    # One colour per type of entity
    assert marks[0][0] != marks[1][0]
    # What is typed is not read as HTML
    assert "une" in html and "&lt;" in html and " < " not in html


@pytest.mark.parametrize("prediction, text", [
    ("negation", "negation"), (["microbiology", "virology"], "microbiology|virology"), ([], ""), (3.14159, "3.1416"), ("ab", "ab"),
])
def test_prediction_as_it_is_written_in_a_file(prediction, text, predict):
    assert predict.as_text(prediction) == text


# ---------------------------------------------------------------------------------------------
# The runs and the models that they kept, read in results written by hand
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def runs(predict, tmp_path, monkeypatch):
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    monkeypatch.setattr(predict, "RECIPES_DIR", str(recipes))
    return Runs(recipes)


def test_runs_are_read_in_their_results(runs, predict):
    path = runs.write(seed=3, validation=0.25)
    [run] = predict.read_runs()
    assert (run.task, run.model, run.seed, run.fewshot) == ("cas-pos", MODEL, 3, 1.0)
    assert (run.validation, run.validation_metric) == (0.25, "f1")
    assert run.path == str(path)
    # The path that the run wrote is the one from the directory of its script
    assert run.model_dir == str(runs.model_dir(path)) and run.missing is None


def test_score_of_the_validation_set_of_the_first_scripts(runs, predict):
    """They did not write it with its name: it is the best score of the training."""
    runs.write(seed=3, validation=0.25, recent=False)
    [run] = predict.read_runs()
    assert (run.validation, run.validation_metric) == (0.25, None)


def test_runs_are_sorted_and_filtered(runs, predict):
    runs.write(task="cas-pos", seed=1)
    runs.write(task="cas-pos", seed=0)
    runs.write(task="cas-cls", seed=0)
    runs.write(task="quaero-ner_emea", seed=0, model="another/model")
    assert [(run.task, run.model, run.seed) for run in predict.read_runs()] == [
        ("cas-cls", MODEL, 0), ("cas-pos", MODEL, 0), ("cas-pos", MODEL, 1), ("quaero-ner_emea", "another/model", 0),
    ]
    assert [(run.task, run.seed) for run in predict.read_runs("cas-pos")] == [("cas-pos", 0), ("cas-pos", 1)]
    assert [run.task for run in predict.read_runs(model="another/model")] == ["quaero-ner_emea"]
    assert predict.read_runs("essai-pos") == []


@pytest.mark.parametrize("recent, offline", [(True, False), (True, True), (False, True), (False, False)])
def test_model_of_a_run_is_found_by_its_name_on_the_hub(recent, offline, runs, predict):
    """In offline mode a run has the copy of its model: the first scripts did not write its name."""
    runs.write(recent=recent, offline=offline)
    runs.write(model="another/model", recent=recent, offline=offline)
    assert len(predict.read_runs(model=MODEL)) == 1
    assert len(predict.read_runs(model="some-org/some-model")) == (1 if offline else 0)


def test_model_which_is_not_there(runs, predict):
    runs.write(seed=0, kept=None)
    runs.write(seed=1, kept=False)
    not_kept, removed = predict.read_runs()
    assert (not_kept.model_dir, not_kept.missing) == (None, "not kept (--keep_model false)")
    assert (removed.model_dir, removed.missing) == (None, "not here")


def test_best_run_is_the_best_one_on_the_validation_set(runs, predict):
    runs.write(seed=0, validation=0.5)
    best = runs.write(seed=1, validation=0.75)
    runs.write(seed=2, validation=0.625)
    # A run that is better, but whose model is not there, or which read a part of the training set
    runs.write(seed=3, validation=0.875, kept=False)
    runs.write(seed=4, validation=0.9375, kept=None)
    runs.write(seed=5, validation=0.96875, fewshot=0.5)
    assert predict.best_run(predict.read_runs()).path == str(best)


def test_first_seed_is_taken_among_equal_runs(runs, predict):
    runs.write(seed=2, validation=0.5)
    first = runs.write(seed=1, validation=0.5)
    assert predict.best_run(predict.read_runs()).path == str(first)


@pytest.mark.parametrize("recent", [True, False])
def test_lowest_error_is_the_best_similarity_run(recent, runs, predict):
    """The error of a similarity score is better when it is lower: the first scripts did not write its name."""
    scores = {"EDRM": 0.5, "spearman_correlation_coef": 0.25, "spearman_correlation_p": 0.01}
    runs.write(task="clister-regr", seed=0, validation=1.5, metric="rmse", metrics=scores, recent=recent)
    best = runs.write(task="clister-regr", seed=1, validation=0.75, metric="rmse", metrics=scores, recent=recent)
    runs.write(task="clister-regr", seed=2, validation=1.25, metric="rmse", metrics=scores, recent=recent)
    read = predict.read_runs()
    assert all(run.lower_is_better for run in read)
    assert predict.best_run(read).path == str(best)
    # The other scores are better when they are higher
    runs.write(task="cas-pos", recent=recent)
    assert not predict.read_runs("cas-pos")[0].lower_is_better


def test_no_best_run_without_a_kept_model(runs, predict):
    runs.write(seed=0, kept=False)
    assert predict.best_run(predict.read_runs()) is None
    assert predict.best_run([]) is None


def test_list_of_the_runs(runs, predict):
    runs.write(seed=0, validation=0.5)
    runs.write(seed=1, validation=0.75)
    runs.write(seed=2, validation=0.875, kept=None)
    runs.write(seed=3, validation=0.9, kept=False)
    runs.write(task="clister-regr", seed=0, validation=1.5, metric="rmse", recent=False,
               metrics={"EDRM": 0.5, "spearman_correlation_coef": 0.25, "spearman_correlation_p": 0.01})
    runs.write(task="cas-cls", seed=0, validation=0.25, metric="accuracy",
               metrics={"macro avg": {"f1-score": 0.125}, "weighted avg": {"f1-score": 0.375}, "accuracy": 0.5})

    lines = predict.list_runs().splitlines()
    table = [re.split(r"  +", line) for line in lines[:lines.index("")]]
    assert table == [
        ["task", "model", "seed", "validation", "test", "kept model"],
        ["cas-cls", MODEL, "0", "0.2500 (accuracy)", "macro_f1 0.1250, weighted_f1 0.3750", "yes, taken"],
        ["cas-pos", MODEL, "0", "0.5000 (f1)", "f1 0.7500, accuracy 0.8750", "yes"],
        ["cas-pos", MODEL, "1", "0.7500 (f1)", "f1 0.7500, accuracy 0.8750", "yes, taken"],
        ["cas-pos", MODEL, "2", "0.8750 (f1)", "f1 0.7500, accuracy 0.8750", "not kept (--keep_model false)"],
        ["cas-pos", MODEL, "3", "0.9000 (f1)", "f1 0.7500, accuracy 0.8750", "not here"],
        # The first scripts did not write the name of the score of the validation set
        ["clister-regr", MODEL, "0", "1.5000", "edrm 0.5000, spearman 0.2500", "yes, taken"],
    ]
    # The columns are aligned
    assert len({line.index(MODEL) for line in lines[1:lines.index("")]}) == 1

    assert {line.split()[0] for line in predict.list_runs("cas-pos").splitlines()[1:4]} == {"cas-pos"}
    assert predict.list_runs("essai-pos") == f"No run in {runs.recipes / 'essai' / 'runs'} for the task essai-pos."
    assert predict.list_runs(model="x/y") == f"No run in {runs.recipes / '*' / 'runs'} for the model x/y."


def test_list_says_which_part_of_the_training_set_a_run_read(runs, predict):
    runs.write(seed=0)
    runs.write(seed=0, fewshot=0.25, validation=0.75)
    lines = predict.list_runs().splitlines()
    assert re.split(r"  +", lines[0]) == ["task", "model", "seed", "train", "validation", "test", "kept model"]
    # The run on a part of the training set is not the one that is taken
    assert [re.split(r"  +", line)[3::3] for line in lines[1:3]] == [["1", "yes, taken"], ["0.25", "yes"]]


def test_run_whose_model_is_used(runs, predict):
    runs.write(seed=0, validation=0.5)
    best = runs.write(seed=1, validation=0.75)
    removed = runs.write(seed=2, validation=0.875, kept=False)
    runs.write(task="cas-cls", seed=7, validation=0.9)

    run, among = predict.chosen_run("cas-pos", MODEL)
    assert run.path == str(best) and [other.seed for other in among] == [0, 1, 2]
    assert predict.chosen_run("cas-pos", MODEL, seed=0)[0].seed == 0

    with pytest.raises(predict.PredictError) as error:
        predict.chosen_run("cas-pos", MODEL, seed=7)
    assert str(error.value) == f"No run of cas-pos for {MODEL} has the seed 7: the seeds of its runs are 0, 1, 2."
    with pytest.raises(predict.PredictError) as error:
        predict.chosen_run("cas-pos", MODEL, seed=2)
    assert str(error.value) == f"The model of the run of cas-pos for {MODEL} with the seed 2 is not here: {removed}"


def test_no_run_for_the_model(runs, predict):
    runs.write(model="another/model")
    with pytest.raises(predict.PredictError) as error:
        predict.chosen_run("cas-pos", MODEL)
    assert str(error.value) == (
        f"No run of cas-pos for {MODEL} in {runs.recipes / 'cas' / 'runs'}.\n"
        "The models that have runs of this task: another/model\n"
        f"A run is computed with `python run.py --tasks cas-pos --models {MODEL}`."
    )
    with pytest.raises(predict.PredictError) as error:
        predict.chosen_run("essai-pos", MODEL)
    assert "No model has a run of this task." in str(error.value)


def test_no_run_kept_its_model(runs, predict):
    not_kept = runs.write(seed=0, kept=None)
    removed = runs.write(seed=1, kept=False)
    with pytest.raises(predict.PredictError) as error:
        predict.chosen_run("cas-pos", MODEL)
    assert str(error.value) == (
        f"No run of cas-pos for {MODEL} kept its model:\n"
        f"  seed 0: not kept (--keep_model false) ({not_kept})\n"
        f"  seed 1: not here ({removed})\n"
        "A run keeps its model unless it is given `--keep_model false`."
    )


def test_runs_on_a_part_of_the_training_set_are_not_used(runs, predict):
    runs.write(seed=0, fewshot=0.5)
    with pytest.raises(predict.PredictError, match="No run of cas-pos"):
        predict.chosen_run("cas-pos", MODEL)


# ---------------------------------------------------------------------------------------------
# The script of a task
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_script_of_a_task_is_the_one_of_its_launcher(task, predict):
    assert predict.script_of(task.name) == str(task.scripts_dir / task.script)


def test_script_named_by_the_description_of_a_model(predict, tmp_path):
    """A model that a run kept names the script of its task: it is the one that is loaded."""
    description = {"task": "cas-pos", "script": "recipes/cas/scripts/task_3_finetuning_bert_ner.py"}
    assert predict.script_for(description) == str(RECIPES / "cas" / "scripts" / "task_3_finetuning_bert_ner.py")
    # The first scripts did not name it: the script is the one of the task
    assert predict.script_for({"task": "cas-pos"}) == str(RECIPES / "cas" / "scripts" / "task_1_finetuning_bert_pos.py")


@pytest.mark.parametrize("named", [
    # A script that is not in the repository, by its path and from the root of the repository
    "{outside}", "{outside_from_the_repository}",
    # Files of the repository that are not scripts of the recipes
    "run.py", "recipes/cas/yaml/pos.yaml", "recipes/cas/scripts/run_task_1.sh",
    "recipes/cas/scripts/nothing.py",
])
def test_script_which_is_not_one_of_the_recipes_is_not_loaded(named, predict, tmp_path):
    """The description of a model comes with the model, maybe from somewhere else: it does not choose the code that is run."""
    outside = tmp_path / "other.py"
    outside.write_text("raise SystemExit('this script was run')\n")
    named = named.format(outside=outside, outside_from_the_repository=os.path.relpath(outside, REPO))
    assert named.startswith(("/", "..")) == ("other.py" in named)
    assert predict.script_for({"task": "cas-pos", "script": named}) == str(RECIPES / "cas" / "scripts" / "task_1_finetuning_bert_pos.py")


def test_unknown_task(predict):
    with pytest.raises(predict.PredictError, match="^cas-nothing is not a task of run.py. Its tasks are: cas-cls, cas-ner_neg, "):
        predict.script_of("cas-nothing")


def test_description_of_a_model_is_the_file_that_the_runs_write(predict, training):
    assert predict.DESCRIPTION == training.DESCRIPTION


def imported_modules(stderr):
    """The modules that a program started with `-X importtime` loaded."""
    return {line.split("|")[-1].strip().split(".")[0] for line in stderr.splitlines() if line.startswith("import time:")}


def test_predict_py_does_not_load_torch_to_list_the_runs():
    """The list of the runs is read in their results: nothing of what trains a model is needed."""
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(REPO / "predict.py"), "--list", "--task", "cas-pos", "--model", "no/model"],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.startswith("No run in ")
    imported = imported_modules(completed.stderr)
    assert "json" in imported and not imported & {"torch", "transformers", "datasets"}


@pytest.mark.parametrize("arguments, message", [
    ([], "give the model: --task and --model, or --model-dir"),
    (["--task", "cas-pos"], "give the model: --task and --model, or --model-dir"),
    (["--model-dir", "somewhere", "--seed", "1"], "--model-dir is the directory of one model"),
    ([], "(the runs are listed by --list)"),
    (["--list", "a sentence"], "--list only goes with --task and --model"),
    (["--list", "--input", "a.txt"], "--list only goes with --task and --model"),
    (["--task", "cas-pos", "--model", MODEL, "--input", "a.txt", "a sentence"], "give texts, or --input"),
    (["--task", "cas-pos", "--model", MODEL, "--input", "a.txt", "--format", "json"], "argument --format: invalid choice: 'json'"),
    (["--list", "--evaluate", "a.conll"], "--list only goes with --task and --model"),
    (["--task", "cas-pos", "--model", MODEL, "--input", "a.txt", "--evaluate", "b.conll"], "give texts, or --input, or --evaluate"),
], ids=lambda value: " ".join(value) if isinstance(value, list) else None)
def test_wrong_arguments(arguments, message):
    completed = subprocess.run([sys.executable, str(REPO / "predict.py"), *arguments], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2
    assert message in completed.stderr


def test_error_is_a_message(tmp_path):
    """What cannot be done is said at once, in a message: without the trace of the program, before torch is loaded."""
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", str(REPO / "predict.py"), "--task", "cas-pos", "--model", "no/such-model", "a sentence"],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )
    assert completed.returncode == 1
    imported = imported_modules(completed.stderr)
    message = [line for line in completed.stderr.splitlines() if not line.startswith("import time:")]
    assert message[0].startswith("No run of cas-pos for no/such-model in ") and len(message) == 3
    assert "json" in imported and not imported & {"torch", "transformers", "datasets"}

    # A directory that is not the one of a model
    completed = subprocess.run(
        [sys.executable, str(REPO / "predict.py"), "--model-dir", str(tmp_path), "a sentence"], capture_output=True, text=True, timeout=120)
    assert (completed.returncode, completed.stderr) == (1, f"{tmp_path} is not the directory of a model: it has no config.json\n")


# ---------------------------------------------------------------------------------------------
# What is typed for a task, with a model that predicts nothing
# ---------------------------------------------------------------------------------------------

def model_of(predict, name, inputs, reference_column, word_labels=False, labels=()):
    task = types.SimpleNamespace(inputs=inputs, reference_column=reference_column, word_labels=word_labels, id_column="id")
    return predict.Model(name, task, types.SimpleNamespace(labels=list(labels)), {})


@pytest.fixture
def tagger(predict):
    return model_of(predict, "cas-pos", {"tokens": "words"}, "pos_tags", word_labels=True)


@pytest.fixture
def classifier(predict):
    return model_of(predict, "cas-cls", {"tokens": "words"}, "label")


@pytest.fixture
def comparer(predict):
    return model_of(predict, "clister-regr", {"text_1": "text", "text_2": "text"}, "label")


def test_example_from_what_is_typed(tagger, comparer, predict):
    assert tagger.example("l'examen est normal.") == {"tokens": ["l'", "examen", "est", "normal", "."]}
    # The words are given as they are
    assert tagger.example("l'examen est normal .", words=True) == {"tokens": ["l'examen", "est", "normal", "."]}
    # A text is not split
    assert comparer.example("l'examen est normal.", "rien") == {"text_1": "l'examen est normal.", "text_2": "rien"}
    with pytest.raises(predict.PredictError) as error:
        comparer.example("one text")
    assert str(error.value) == "clister-regr reads 2 texts (text_1, text_2): 1 given"
    with pytest.raises(predict.PredictError, match=r"cas-pos reads 1 text \(tokens\): 2 given"):
        tagger.example("one", "two")


def test_examples_of_the_texts_of_the_command_line(classifier, comparer, predict):
    """One text per example, or as many as the task reads."""
    def examples(model, *texts):
        return predict.typed_examples(model, types.SimpleNamespace(texts=list(texts), words=False))

    assert examples(classifier, "une toux", "pas de fièvre") == [{"tokens": ["une", "toux"]}, {"tokens": ["pas", "de", "fièvre"]}]
    assert examples(comparer, "a", "b", "c", "d") == [{"text_1": "a", "text_2": "b"}, {"text_1": "c", "text_2": "d"}]
    with pytest.raises(predict.PredictError) as error:
        examples(comparer, "a", "b", "c")
    assert str(error.value) == "clister-regr reads 2 texts for an example (text_1, text_2): 3 given"


def test_prediction_as_it_is_shown(tagger, classifier, comparer, predict):
    assert predict.shown(tagger, {"tokens": ["une", "toux"]}, ["O", "B-DISO"]) == "une toux/DISO"
    assert predict.shown(classifier, {"tokens": ["une", "toux"]}, "neutral") == "neutral"
    # The labels of a text, the letters of the right answers and a similarity score
    assert predict.shown(classifier, {}, ["microbiology", "virology"]) == "microbiology, virology"
    assert predict.shown(classifier, {}, []) == "(no label)"
    assert predict.shown(classifier, {}, "ab") == "ab"
    assert predict.shown(comparer, {}, 3.14159) == "3.14"


# ---------------------------------------------------------------------------------------------
# The files that are read and written
# ---------------------------------------------------------------------------------------------

def test_text_file(tagger, comparer, predict, tmp_path):
    path = tmp_path / "sentences.txt"
    path.write_text("Le patient tousse.\n\n  \nl'examen est normal\r\n", encoding="utf-8")
    read = predict.read_examples(str(path), tagger)
    assert read.examples == [{"tokens": ["Le", "patient", "tousse", "."]}, {"tokens": ["l'", "examen", "est", "normal"]}]
    assert read.columns == ["tokens"]
    # The words are given as they are
    assert predict.read_examples(str(path), tagger, words=True).examples[0] == {"tokens": ["Le", "patient", "tousse."]}

    pairs = tmp_path / "pairs.txt"
    pairs.write_text("une toux\tune fièvre\nseule\n", encoding="utf-8")
    with pytest.raises(predict.PredictError) as error:
        predict.read_examples(str(pairs), comparer)
    assert str(error.value) == (
        f"{pairs}, line 2: clister-regr reads 2 texts (text_1, text_2): 1 given. The texts of an example are separated by tabulations.")
    pairs.write_text("une toux\tune fièvre\n", encoding="utf-8")
    assert predict.read_examples(str(pairs), comparer).examples == [{"text_1": "une toux", "text_2": "une fièvre"}]


CONLL = "-DOCSTART- -X- O\n\nLe\tx\tB-DET\npatient\tx\tB-NOM\n\n\ntousse B-VER\n. B-SENT\n"


def test_conll_file(tagger, predict, tmp_path):
    """One word per line, which starts with it, and an empty line after a sentence."""
    path = tmp_path / "corpus.conll"
    path.write_text(CONLL, encoding="utf-8")
    read = predict.read_examples(str(path), tagger)
    assert read.examples == [{"tokens": ["Le", "patient"]}, {"tokens": ["tousse", "."]}]
    assert read.columns == ["tokens"]

    # A file which is not named as a CoNLL file is a text file, unless its format is given
    other = tmp_path / "words.txt"
    other.write_text("Le\npatient\n", encoding="utf-8")
    assert predict.read_examples(str(other), tagger).examples == [{"tokens": ["Le"]}, {"tokens": ["patient"]}]
    assert predict.read_examples(str(other), tagger, file_format="conll").examples == [{"tokens": ["Le", "patient"]}]


def write_table(path, rows, delimiter=","):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter=delimiter).writerows(rows)


@pytest.mark.parametrize("name, delimiter", [("examples.csv", ","), ("examples.tsv", "\t")])
def test_table_file(name, delimiter, classifier, predict, tmp_path):
    path = tmp_path / name
    write_table(path, [["patient", "tokens"], ["p1", "pas de toux, ni fièvre"], ["p2", "une toux"]], delimiter)
    read = predict.read_examples(str(path), classifier)
    assert read.examples == [{"patient": "p1", "tokens": ["pas", "de", "toux", ",", "ni", "fièvre"]}, {"patient": "p2", "tokens": ["une", "toux"]}]
    # The columns of the file are written again with the predictions
    assert read.columns == ["patient", "tokens"]
    assert predict.read_examples(str(path), classifier, words=True).examples[0]["tokens"] == ["pas", "de", "toux,", "ni", "fièvre"]


def test_table_file_without_a_column(classifier, comparer, predict, tmp_path):
    path = tmp_path / "examples.csv"
    path.write_text("tokens,class\nune toux,neutral\n", encoding="utf-8")
    assert len(predict.read_examples(str(path), classifier).examples) == 1
    with pytest.raises(predict.PredictError) as error:
        predict.read_examples(str(path), comparer)
    assert str(error.value) == (
        f"{path} has no column text_1, text_2: its first line names its columns, which are tokens, class. "
        "clister-regr reads the columns text_1, text_2.")


def test_file_that_a_task_does_not_read(tagger, classifier, predict, tmp_path):
    for name in ("examples.csv", "examples.conll"):
        (tmp_path / name).write_text("tokens\nune toux\n", encoding="utf-8")
    with pytest.raises(predict.PredictError, match="cas-cls does not label words, its examples are not read in a CoNLL file"):
        predict.read_examples(str(tmp_path / "examples.conll"), classifier)
    with pytest.raises(predict.PredictError, match="cas-pos labels words, its examples are read in a CoNLL file"):
        predict.read_examples(str(tmp_path / "examples.csv"), tagger)
    with pytest.raises(predict.PredictError, match="nothing.txt does not exist"):
        predict.read_examples(str(tmp_path / "nothing.txt"), classifier)


def written(predict, model, read, predictions, **options):
    output = io.StringIO()
    predict.write_predictions(output, model, read, predictions, **options)
    return output.getvalue()


def test_labels_are_written_in_the_conll_format(tagger, predict):
    read = predict.Examples([{"tokens": ["Le", "patient"]}, {"tokens": ["tousse"]}], ["tokens"])
    assert written(predict, tagger, read, [["B-DET", "B-NOM"], ["B-VER"]]) == "Le\tB-DET\npatient\tB-NOM\n\ntousse\tB-VER\n\n"


def test_classes_are_written_in_csv(classifier, comparer, predict):
    read = predict.Examples(
        [{"patient": "p1", "tokens": ["pas", "de", "toux"]}, {"patient": "p2", "tokens": ["une", '"toux"']}], ["patient", "tokens"])
    assert list(csv.reader(io.StringIO(written(predict, classifier, read, ["negation", "neutral"])))) == [
        ["patient", "tokens", "prediction"], ["p1", "pas de toux", "negation"], ["p2", 'une "toux"', "neutral"],
    ]
    # Several labels, and a score
    assert written(predict, classifier, read, [["a", "b"], []]).splitlines()[1:] == ["p1,pas de toux,a|b", 'p2,"une ""toux""",']
    read = predict.Examples([{"text_1": "a, b", "text_2": "c"}], ["text_1", "text_2"])
    assert written(predict, comparer, read, [2.5]) == 'text_1,text_2,prediction\n"a, b",c,2.5000\n'


# ---------------------------------------------------------------------------------------------
# The annotated files, and what a model should predict for their examples
# ---------------------------------------------------------------------------------------------

def test_annotated_conll_file(tagger, predict, tmp_path):
    """The label of a word is at the end of its line."""
    path = tmp_path / "corpus.conll"
    path.write_text(CONLL, encoding="utf-8")
    read = predict.read_examples(str(path), tagger, annotated=True)
    assert read.examples == [
        {"tokens": ["Le", "patient"], "pos_tags": ["B-DET", "B-NOM"]}, {"tokens": ["tousse", "."], "pos_tags": ["B-VER", "B-SENT"]},
    ]
    assert read.columns == ["tokens", "pos_tags"]

    path.write_text("Le\tB-DET\npatient\n", encoding="utf-8")
    with pytest.raises(predict.PredictError, match=r"corpus.conll, line 2: a word and its label are expected"):
        predict.read_examples(str(path), tagger, annotated=True)
    # Without labels, the file is still one of words
    assert predict.read_examples(str(path), tagger).examples == [{"tokens": ["Le", "patient"]}]


def test_annotated_table_file(classifier, predict, tmp_path):
    path = tmp_path / "examples.csv"
    write_table(path, [["tokens", "label"], ["pas de toux", "negation"]])
    read = predict.read_examples(str(path), classifier, annotated=True)
    assert read.examples == [{"tokens": ["pas", "de", "toux"], "label": "negation"}]

    # The column of the answers has the name it has in the dataset of the task
    write_table(path, [["tokens", "class"], ["pas de toux", "negation"]])
    with pytest.raises(predict.PredictError) as error:
        predict.read_examples(str(path), classifier, annotated=True)
    assert str(error.value) == (
        f"{path} has no column label: its first line names its columns, which are tokens, class. cas-cls reads the columns tokens, label.")


def test_annotated_file_that_a_task_does_not_read(tagger, classifier, predict, tmp_path):
    path = tmp_path / "examples.txt"
    path.write_text("une toux\n", encoding="utf-8")
    with pytest.raises(predict.PredictError, match="cas-pos labels words, its annotated examples are read in a CoNLL file"):
        predict.read_examples(str(path), tagger, annotated=True)
    with pytest.raises(predict.PredictError, match="the annotated examples of cas-cls are read in a CSV or TSV file .* whose columns are tokens, label"):
        predict.read_examples(str(path), classifier, annotated=True)


def test_expected_labels_are_written_before_the_predicted_ones(tagger, classifier, predict):
    read = predict.Examples([{"tokens": ["Le", "patient"]}, {"tokens": ["tousse"]}], ["tokens"])
    text = written(predict, tagger, read, [["B-DET", "B-NOM"], ["B-VER"]], references=[["B-DET", "B-ADJ"], ["B-VER"]])
    assert text == "Le\tB-DET\tB-DET\npatient\tB-ADJ\tB-NOM\n\ntousse\tB-VER\tB-VER\n\n"

    # The classes that are expected are written as the model names them: here the dataset has their numbers
    read = predict.Examples([{"tokens": ["pas", "de", "toux"], "label": 1}, {"tokens": ["une", "toux"], "label": 3}], ["tokens", "label"])
    text = written(predict, classifier, read, ["negation", "neutral"], references=["negation", "speculation"])
    assert text.splitlines() == ["tokens,label,prediction", "pas de toux,negation,negation", "une toux,speculation,neutral"]


def test_scores_are_shown_with_the_names_of_the_classes(predict):
    metrics = {
        "0": {"precision": 0.5, "recall": 1.0, "f1-score": 0.75, "support": 2.0},
        "accuracy": 0.5,
        "macro avg": {"precision": 0.25, "recall": 0.5, "f1-score": 0.375, "support": 4.0},
    }
    assert predict.scores_as_text(metrics, ["negation", "neutral"]).splitlines() == [
        "accuracy: 0.5000",
        "  negation: precision 0.5000, recall 1.0000, f1-score 0.7500, support 2",
        "  macro avg: precision 0.2500, recall 0.5000, f1-score 0.3750, support 4",
    ]
    entities = {"DISO": {"precision": 0.5, "recall": 0.25, "f1": 0.375, "number": 8}, "overall_f1": 0.375}
    assert predict.scores_as_text(entities, ["O", "B-DISO"]).splitlines() == [
        "overall_f1: 0.3750", "  DISO: precision 0.5000, recall 0.2500, f1 0.3750, number 8"]


def task_of(predict, name):
    return predict.load_task(predict.script_of(name))


def test_expected_class_of_an_example(predict):
    from datasets import ClassLabel, Features

    task = task_of(predict, "cas-cls")
    labels = ["negation_speculation", "negation", "neutral", "speculation"]
    # In a file, the name of the class
    assert task.reference({"label": "neutral"}, labels) == "neutral"
    # In a dataset, its number: the dataset names its classes, in its own order
    assert task.reference({"label": 2}, labels) == "neutral"
    features = Features({"label": ClassLabel(names=["neutral", "negation"])})
    assert task.reference({"label": 1}, labels, features) == "negation"
    for unknown in ("neutre", "2", 4, -1):
        with pytest.raises(ValueError) as error:
            task.reference({"label": unknown}, labels)
        assert str(error.value) == f"Unknown class {unknown!r}: the classes are 'negation_speculation', 'negation', 'neutral', 'speculation'"

    # Classes that are named by numbers: a text is a name, not the number of a class
    task = task_of(predict, "deft2020-cls")
    assert task.reference({"correct_cible": "1"}, ["1", "2", "3"]) == "1"
    assert task.reference({"correct_cible": 1}, ["1", "2", "3"]) == "2"


def test_expected_labels_of_a_sentence(predict):
    from datasets import ClassLabel, Features, Sequence

    task = task_of(predict, "cas-pos")
    labels = ["B-DET", "B-NOM", "B-VER"]
    assert task.reference({"pos_tags": ["B-NOM", "B-VER"]}, labels) == ["B-NOM", "B-VER"]
    features = Features({"pos_tags": Sequence(ClassLabel(names=["B-VER", "B-NOM"]))})
    assert task.reference({"pos_tags": [0, 1, 1]}, labels, features) == ["B-VER", "B-NOM", "B-NOM"]
    with pytest.raises(ValueError, match="Unknown class 'B-ADJ'"):
        task.reference({"pos_tags": ["B-NOM", "B-ADJ"]}, labels)


def test_expected_labels_of_a_text_are_read_as_they_are_written(predict):
    from datasets import ClassLabel, Features, Sequence

    task = task_of(predict, "morfitt-cls")
    labels = ["microbiology", "etiology", "virology"]
    # In a file: what `predict.py` writes for several labels
    several = predict.as_text(["virology", "microbiology"])
    assert task.reference({"specialities": several}, labels) == ["virology", "microbiology"]
    assert task.reference({"specialities": " etiology | virology "}, labels) == ["etiology", "virology"]
    assert task.reference({"specialities": ""}, labels) == []
    with pytest.raises(ValueError, match="Unknown class 'cardiology'"):
        task.reference({"specialities": "virology|cardiology"}, labels)
    # In a dataset: the numbers of the labels, or one number for each label of the model
    features = Features({"specialities": Sequence(ClassLabel(names=["virology", "microbiology"]))})
    assert task.reference({"specialities": [1, 0]}, labels, features) == ["microbiology", "virology"]
    assert task.reference({"specialities_one_hot": [1.0, 0.0, 1.0]}, labels) == ["microbiology", "virology"]


def test_expected_answers_and_score(predict):
    from datasets import ClassLabel, Features, Sequence

    task = task_of(predict, "frenchmedmcqa-mcqa")
    # The letters of the right answers, in a file, whatever is between them
    for answers in ("ab", "ba", "a, b", "A|B", "b a"):
        assert task.reference({"correct_answers": answers}, task.labels) == "ab"
    # In a dataset: their numbers
    features = Features({"correct_answers": Sequence(ClassLabel(names=["a", "b", "c", "d", "e"]))})
    assert task.reference({"correct_answers": [4, 0, 2]}, task.labels, features) == "ace"
    with pytest.raises(ValueError, match="Unknown class 'f'"):
        task.reference({"correct_answers": "af"}, task.labels)

    task = task_of(predict, "clister-regr")
    assert task.reference({"label": "2.5"}, None) == task.reference({"label": 2.5}, None) == 2.5


def test_scores_of_predictions(predict):
    """The scores of a task for what is expected and what is predicted, as `predict.py` gives them: by their names."""
    labels = ["negation_speculation", "negation", "neutral", "speculation"]
    scores = task_of(predict, "cas-cls").metrics(["negation", "neutral", "neutral", "negation"], ["negation", "neutral", "negation", "negation"], labels)
    # The scores of a class have the number of the class
    assert scores["accuracy"] == 0.75 and scores["1"]["recall"] == 1.0 and scores["2"]["recall"] == 0.5

    scores = task_of(predict, "morfitt-cls").metrics([["a"], ["a", "b"]], [["a"], ["b"]], ["a", "b", "c"])
    assert (scores["0"]["recall"], scores["1"]["recall"], scores["micro avg"]["precision"]) == (0.5, 1.0, 1.0)

    scores = task_of(predict, "cas-pos").metrics([["B-NOM", "O"], ["B-VER"]], [["B-NOM", "O"], ["B-NOM"]], ["O", "B-NOM", "B-VER"])
    assert scores["overall_accuracy"] == pytest.approx(2 / 3) and scores["NOM"]["precision"] == 0.5

    scores = task_of(predict, "clister-regr").metrics([0.0, 2.5, 5.0], [0.0, 2.5, 5.0], None)
    assert scores["EDRM"] == 1.0 and scores["spearman_correlation_coef"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# The models that the recipes keep after a training on synthetic data
# ---------------------------------------------------------------------------------------------

def the_test_set(task, data):
    """The test set of a task, as its recipe reads it."""
    from datasets import load_from_disk

    return load_from_disk(str(data / task.recipe / f"local_hf_{task.subset}"))["test"]


def close(first, second):
    """Whether two predictions, or two sets of scores, are the same: the numbers of a model depend on how it is run."""
    if isinstance(first, dict):
        return set(first) == set(second) and all(close(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(close(a, b) for a, b in zip(first, second))
    if isinstance(first, float) or isinstance(second, float):
        return (math.isnan(first) and math.isnan(second)) or math.isclose(first, second, rel_tol=1e-3, abs_tol=1e-3)
    return first == second


def named(task, predictions, labels):
    """The predictions that a run wrote, in the form of the ones of `predict.py`: the classes by their names."""
    if task.kind != "cls":
        return predictions
    if isinstance(predictions[0], list):
        return [[label for label, value in zip(labels, vector) if value] for vector in predictions]
    return [labels[index] for index in predictions]


@pytest.mark.e2e
@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_prediction_is_the_one_of_the_run(task, run_recipe, tiny_bert, synthetic_data, predict):
    """What a model reads is built in the same way to train it and to predict.

    `predict.py` loads the model that the run kept and reads the test set again: its
    predictions are the ones that the run wrote.
    """
    result = run_recipe(task.name, tiny_bert).result
    model = predict.load(model_dir=result["model_name"])
    predictions = model.predict(list(the_test_set(task, synthetic_data)))
    assert close(predictions, named(task, result["predictions"]["system_predictions"], model.labels))


@pytest.mark.e2e
@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_scores_of_the_test_set_are_the_ones_of_the_run(task, run_recipe, tiny_bert, synthetic_data, predict):
    """The test set of a task is annotated data: what is expected for its examples and their scores are the ones of the run."""
    result = run_recipe(task.name, tiny_bert).result
    model = predict.load(model_dir=result["model_name"])
    read = predict.read_examples(str(synthetic_data / task.recipe / f"local_hf_{task.subset}"), model, annotated=True)
    references, identifiers = predict.references_of(model, read)
    assert identifiers == result["predictions"]["identifiers"]
    assert close(references, named(task, result["predictions"]["real_labels"], model.labels))

    predictions = model.predict(read.examples)
    scores = json.loads(json.dumps(model.task.metrics(references, predictions, model.labels, identifiers), default=float))
    assert close(scores, result["metrics"])


@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "cas-cls", "frenchmedmcqa-mcqa"])
def test_prediction_is_the_one_of_the_run_with_the_tokenizer_of_drbert(task_name, run_recipe, tiny_drbert, synthetic_data, predict):
    task = Task(task_name)
    result = run_recipe(task.name, tiny_drbert).result
    model = predict.load(model_dir=result["model_name"])
    predictions = model.predict(list(the_test_set(task, synthetic_data)))
    assert close(predictions, named(task, result["predictions"]["system_predictions"], model.labels))


@pytest.mark.e2e
def test_prediction_is_the_one_of_the_run_with_the_tokenizer_of_flaubert(run_recipe, tiny_flaubert, synthetic_data, predict):
    """The classification of a sentence: the tagging tasks do not write results with FlauBERT, a known defect."""
    task = Task("cas-cls")
    result = run_recipe(task.name, tiny_flaubert).result
    model = predict.load(model_dir=result["model_name"])
    assert "flaubert" in model.description["model"]
    predictions = model.predict(list(the_test_set(task, synthetic_data)))
    assert predictions == named(task, result["predictions"]["system_predictions"], model.labels)


@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "cas-cls"])
def test_text_is_cut_as_when_the_model_was_trained(task_name, run_recipe, tiny_bert, synthetic_data, predict):
    """A text that is too long is cut at the length of the run, which is written with its model."""
    task = Task(task_name)
    result = run_recipe(task.name, tiny_bert, "--max_position_embeddings", "6").result
    model = predict.load(model_dir=result["model_name"])
    assert model.predictor.max_length == 6
    examples = list(the_test_set(task, synthetic_data))
    predictions = model.predict(examples)
    assert predictions == named(task, result["predictions"]["system_predictions"], model.labels)
    if task_name == "cas-pos":
        # Four sub-tokens are read between the two that the tokenizer adds: the next words have no label
        assert all(len(example["tokens"]) > 4 for example in examples)
        assert all(prediction[4:] == ["O"] * (len(prediction) - 4) for prediction in predictions)
        assert all("O" not in prediction[:3] for prediction in predictions)


@pytest.fixture(scope="session")
def token_classification(predict):
    """The module of the tasks that label words, as `predict.py` loads it."""
    predict.shared_code()
    from common import token_classification
    return token_classification


def kept_flaubert_tagger(predict, token_classification, tiny_flaubert, directory, **options):
    """A model of FlauBERT for a tagging task, saved as a run saves it: it was not trained."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    training = predict.shared_code()
    model = AutoModelForTokenClassification.from_pretrained(tiny_flaubert, num_labels=3)
    training.name_outputs(model, ["B-DISO", "I-DISO", "O"])
    model.save_pretrained(directory)
    AutoTokenizer.from_pretrained(tiny_flaubert).save_pretrained(directory)
    return training.Predictor(token_classification.Task(dataset="DrBenchmark/QUAERO"), directory, **options)


def test_flaubert_gives_one_label_per_word(predict, token_classification, tiny_flaubert, tmp_path):
    predictor = kept_flaubert_tagger(predict, token_classification, tiny_flaubert, tmp_path / "kept", model_name="flaubert/flaubert_base_cased")
    assert predictor.labels == ["B-DISO", "I-DISO", "O"]
    examples = [{"tokens": ["le", "patient", "tousse"]}, {"tokens": ["toux"]}, {"tokens": []}]
    predictions = predictor(examples)
    assert [len(prediction) for prediction in predictions] == [3, 1, 0]
    assert {label for prediction in predictions for label in prediction} <= {"O", "B-DISO", "I-DISO"}


def test_words_that_a_model_does_not_read_have_no_entity(predict, token_classification, tiny_flaubert, tmp_path):
    """A sentence which is cut: `O` for the words after the cut, as in the predictions that a run writes."""
    # The name of the model that was fine-tuned is not known: the tokenizer says that it is FlauBERT
    predictor = kept_flaubert_tagger(predict, token_classification, tiny_flaubert, tmp_path / "kept", max_length=4)
    assert predictor.model_name == "FlaubertTokenizer"
    # Two tokens are added: the two letters of the first word are the only ones that are read
    [prediction] = predictor([{"tokens": ["le", "patient", "tousse"]}])
    assert prediction[1:] == ["O", "O"]

    logits = [[9, 0, 0], [0, 9, 0], [9, 0, 0], [0, 0, 9]]
    encoded = {"word_ids": [None, 0, 0, None], "words": 3}
    # The label of a word is the one of its first sub-token
    assert token_classification.Task(dataset="x").decode(logits, encoded, ["B-DISO", "I-DISO", "O"]) == ["I-DISO", "O", "O"]


@pytest.mark.e2e
@pytest.mark.parametrize("task_name, named_classes", [("cas-pos", True), ("frenchmedmcqa-mcqa", True), ("cas-cls", False), ("clister-regr", True)])
def test_model_that_the_first_scripts_kept(task_name, named_classes, run_recipe, tiny_bert, synthetic_data, predict, tmp_path):
    """The first scripts saved a model without what describes it, and only the tagging models with the names of their classes."""
    task = Task(task_name)
    result = run_recipe(task.name, tiny_bert).result
    recent = predict.load(model_dir=result["model_name"])
    kept = tmp_path / "kept_best_model"
    shutil.copytree(result["model_name"], kept)
    (kept / "drbenchmark.json").unlink()
    (kept / "README.md").unlink()
    if task.kind not in ("pos", "ner"):
        config = read_json(kept / "config.json")
        config["id2label"] = {str(index): f"LABEL_{index}" for index in range(len(config["id2label"]))}
        config["label2id"] = {name: int(index) for index, name in config["id2label"].items()}
        (kept / "config.json").write_text(json.dumps(config))

    with pytest.raises(predict.PredictError, match="has no drbenchmark.json, which says what the task of the model is"):
        predict.load(model_dir=str(kept))
    model = predict.load(task=task.name, model_dir=str(kept))
    # The classes of the question answering task do not depend on its dataset: the task names them
    assert (model.labels == recent.labels) is named_classes
    examples = list(the_test_set(task, synthetic_data))
    if named_classes:
        assert close(model.predict(examples), recent.predict(examples))
    else:
        assert model.labels == ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"]
        assert [recent.labels[int(label.split("_")[1])] for label in model.predict(examples)] == recent.predict(examples)


# ---------------------------------------------------------------------------------------------
# The command, in a repository where `run.py` computed runs
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained_repository(tmp_path_factory, synthetic_data, tiny_bert):
    """A small copy of the repository, where `run.py` did two runs of two tasks of CAS on the synthetic data."""
    repository = tmp_path_factory.mktemp("trained") / "repository"
    recipe = repository / "recipes" / "cas"
    recipe.mkdir(parents=True)
    for name in ("run.py", "predict.py", "config.yaml", "predict_demo.ipynb"):
        shutil.copy(REPO / name, repository)
    not_copied = shutil.ignore_patterns("__pycache__", "logs")
    # The recipes load their metrics from the root of the repository
    shutil.copytree(REPO / "metrics", repository / "metrics", ignore=not_copied)
    shutil.copytree(RECIPES / "common", repository / "recipes" / "common", ignore=not_copied)
    for name in ("scripts", "yaml"):
        shutil.copytree(RECIPES / "cas" / name, recipe / name, ignore=not_copied)
    # Where `download_datasets_locally.py` saves the datasets: `data/local_hf_<subset>/`
    os.symlink(synthetic_data / "cas", recipe / "data")
    (recipe / "runs").mkdir()

    command = [
        sys.executable, repository / "run.py", "--tasks", "cas-pos", "cas-cls", "--models", tiny_bert, "--nb-run", "2",
        "--epochs", "1", "--offline", "true",
    ]
    completed = run_command(" ".join(shlex.quote(str(argument)) for argument in command), cwd=repository)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
    assert "INFO::4 run(s): 4 computed, 0 already computed, 0 failed" in completed.stderr
    return repository


def run_predict(repository, *arguments, cwd=None, stdin=None):
    return subprocess.run(
        [sys.executable, str(repository / "predict.py"), *(str(argument) for argument in arguments)],
        cwd=cwd or repository, input=stdin, capture_output=True, text=True, timeout=300,
    )


def runs_of(repository, task_name):
    """The results of the runs of a task: {seed: results}."""
    results = [read_json(path) for path in (repository / "recipes" / "cas" / "runs").glob("*.json")]
    return {result["hyperparameters"]["seed"]: result for result in results if result["hyperparameters"]["task"] == task_name}


def taken_run(repository, task_name):
    """The seed and the results of the run whose model `predict.py` takes for a task."""
    runs = runs_of(repository, task_name)
    seed = min(runs, key=lambda seed: (-runs[seed]["validation"]["score"], seed))
    return seed, runs[seed]


@pytest.mark.e2e
def test_list_of_the_models_that_run_py_trained(trained_repository, tiny_bert, tmp_path):
    completed = run_predict(trained_repository, "--list", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    rows = [re.split(r"  +", line) for line in lines[1:5]]
    assert [row[:3] for row in rows] == [
        ["cas-cls", str(tiny_bert), "0"], ["cas-cls", str(tiny_bert), "1"], ["cas-pos", str(tiny_bert), "0"], ["cas-pos", str(tiny_bert), "1"],
    ]
    for task_name, task_rows in (("cas-cls", rows[:2]), ("cas-pos", rows[2:])):
        runs = runs_of(trained_repository, task_name)
        for row in task_rows:
            seed = int(row[2])
            metric = "accuracy" if task_name == "cas-cls" else "f1"
            assert row[3] == f"{runs[seed]['validation']['score']:.4f} ({metric})"
            assert row[5] == ("yes, taken" if seed == taken_run(trained_repository, task_name)[0] else "yes")
    assert rows[2][4] == "f1 {overall_f1:.4f}, accuracy {overall_accuracy:.4f}".format(**runs_of(trained_repository, "cas-pos")[0]["metrics"])


@pytest.mark.e2e
def test_sentence_is_labelled_by_the_model_that_a_run_kept(trained_repository, tiny_bert, tmp_path):
    """The command of the documentation, from another directory: the model is found from the task and the fine-tuned model."""
    completed = run_predict(
        trained_repository, "--task", "cas-pos", "--model", tiny_bert, "le patient présente une douleur thoracique.", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    [line] = completed.stdout.splitlines()
    words = line.split()
    assert [word.rsplit("/", 1)[0] for word in words] == ["le", "patient", "présente", "une", "douleur", "thoracique", "."]
    assert all(word.rsplit("/", 1)[1] in [tag[2:] for tag in synthetic.POS_TAGS] for word in words)

    seed, run = taken_run(trained_repository, "cas-pos")
    assert (f"Model: tiny-bert fine-tuned on cas-pos, seed {seed} (validation f1: {run['validation']['score']:.4f}, "
            "the best of the 2 runs that kept their model)") in completed.stderr

    # The model of the other run
    other = run_predict(trained_repository, "--task", "cas-pos", "--model", tiny_bert, "--seed", 1 - seed, "le patient présente une douleur thoracique.")
    assert other.returncode == 0, other.stderr[-3000:]
    score = runs_of(trained_repository, "cas-pos")[1 - seed]["validation"]["score"]
    assert f"Model: tiny-bert fine-tuned on cas-pos, seed {1 - seed} (validation f1: {score:.4f})" in other.stderr

    # Several sentences
    several = run_predict(trained_repository, "--task", "cas-pos", "--model", tiny_bert, "le patient", "une douleur")
    assert [[word.rsplit("/", 1)[0] for word in line.split()] for line in several.stdout.splitlines()] == [["le", "patient"], ["une", "douleur"]]


@pytest.mark.e2e
def test_sentences_are_asked_for_without_one(trained_repository, tiny_bert):
    completed = run_predict(
        trained_repository, "--task", "cas-cls", "--model", tiny_bert, stdin="le patient présente une douleur\npas de fièvre\n\nnot read\n")
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    assert len(lines) == 2 and set(lines) <= set(synthetic.NEGATION_CLASSES)


def predicted_by_the_run(repository, task_name):
    return taken_run(repository, task_name)[1]["predictions"]["system_predictions"]


@pytest.mark.e2e
def test_file_is_labelled_in_the_conll_format(trained_repository, tiny_bert, synthetic_data, tmp_path):
    test = the_test_set(Task("cas-pos"), synthetic_data)
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("".join(" ".join(example["tokens"]) + "\n" for example in test), encoding="utf-8")
    words = tmp_path / "words.conll"
    words.write_text("".join("".join(f"{word}\n" for word in example["tokens"]) + "\n" for example in test), encoding="utf-8")

    # The sentences of a text file, on the standard output
    completed = run_predict(trained_repository, "--task", "cas-pos", "--model", tiny_bert, "--input", sentences, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    labelled = [[line.split("\t") for line in sentence.splitlines()] for sentence in completed.stdout.strip("\n").split("\n\n")]
    assert [[word for word, _ in sentence] for sentence in labelled] == [example["tokens"] for example in test]
    # The labels are the ones that the run wrote for its test set
    assert [[label for _, label in sentence] for sentence in labelled] == predicted_by_the_run(trained_repository, "cas-pos")

    # The words of a CoNLL file, in a file
    output = tmp_path / "labels.conll"
    in_a_file = run_predict(
        trained_repository, "--task", "cas-pos", "--model", tiny_bert, "--input", words, "--output", output, cwd=tmp_path)
    assert in_a_file.returncode == 0, in_a_file.stderr[-3000:]
    assert in_a_file.stdout == "" and output.read_text(encoding="utf-8") == completed.stdout


@pytest.mark.e2e
def test_classes_of_a_file_are_written_in_csv(trained_repository, tiny_bert, synthetic_data, tmp_path):
    test = the_test_set(Task("cas-cls"), synthetic_data)
    names = test.features["label"].names
    predicted = [names[index] for index in predicted_by_the_run(trained_repository, "cas-cls")]

    # The sentences of a table, whose words are already apart: its columns are written again
    table = tmp_path / "sentences.csv"
    write_table(table, [["id", "tokens"]] + [[example["id"], " ".join(example["tokens"])] for example in test])
    completed = run_predict(trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--input", table, "--words", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    assert rows[0] == ["id", "tokens", "prediction"]
    assert [row[:2] for row in rows[1:]] == [[example["id"], " ".join(example["tokens"])] for example in test]
    assert [row[2] for row in rows[1:]] == predicted

    # The sentences of a text file
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("".join(" ".join(example["tokens"]) + "\n" for example in test), encoding="utf-8")
    completed = run_predict(trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--input", sentences, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    rows = list(csv.reader(io.StringIO(completed.stdout)))
    assert rows[0] == ["tokens", "prediction"] and [row[1] for row in rows[1:]] == predicted

    # A file without any example
    sentences.write_text("\n", encoding="utf-8")
    completed = run_predict(trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--input", sentences, cwd=tmp_path)
    assert (completed.returncode, completed.stderr.splitlines()[-1]) == (1, f"{sentences} has no example")


@pytest.mark.e2e
def test_annotated_corpus_is_scored_with_the_metrics_of_the_task(trained_repository, tiny_bert, synthetic_data, tmp_path):
    """A corpus annotated as the dataset of the task: here its test set, whose scores are the ones of the run."""
    test = the_test_set(Task("cas-pos"), synthetic_data)
    names = test.features["pos_tags"].feature.names
    corpus = tmp_path / "corpus.conll"
    corpus.write_text("".join(
        "".join(f"{word}\t{names[tag]}\n" for word, tag in zip(example["tokens"], example["pos_tags"])) + "\n" for example in test
    ), encoding="utf-8")
    _, run = taken_run(trained_repository, "cas-pos")
    output = tmp_path / "compared.conll"

    completed = run_predict(
        trained_repository, "--task", "cas-pos", "--model", tiny_bert, "--evaluate", corpus, "--output", output, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    assert lines[0] == f"6 examples of {corpus}"
    for name in ("overall_precision", "overall_recall", "overall_f1", "overall_accuracy"):
        assert f"{name}: {run['metrics'][name]:.4f}" in lines
    assert any(re.fullmatch(r"  NOM: precision \d\.\d{4}, recall \d\.\d{4}, f1 \d\.\d{4}, number \d+", line) for line in lines)

    # The file has, for each word, what was expected and what the model predicts
    compared = [line.split("\t") for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert [label for _, label, _ in compared] == [label for sentence in run["predictions"]["real_labels"] for label in sentence]
    assert [label for _, _, label in compared] == [label for sentence in run["predictions"]["system_predictions"] for label in sentence]

    # Without a file for them, the predictions are not written
    completed = run_predict(trained_repository, "--task", "cas-pos", "--model", tiny_bert, "--evaluate", corpus, cwd=tmp_path)
    assert completed.stdout.splitlines() == lines


@pytest.mark.e2e
def test_annotated_table_and_saved_dataset_are_scored(trained_repository, tiny_bert, synthetic_data, tmp_path):
    test = the_test_set(Task("cas-cls"), synthetic_data)
    names = test.features["label"].names
    _, run = taken_run(trained_repository, "cas-cls")
    annotated = tmp_path / "annotated.csv"
    write_table(annotated, [["tokens", "label"]] + [[" ".join(example["tokens"]), names[example["label"]]] for example in test])

    for corpus in (annotated, synthetic_data / "cas" / "local_hf_cls"):
        output = tmp_path / "compared.csv"
        completed = run_predict(
            trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--evaluate", corpus, "--words", "--output", output, cwd=tmp_path)
        assert completed.returncode == 0, completed.stderr[-3000:]
        lines = completed.stdout.splitlines()
        assert lines[:2] == [f"6 examples of {corpus}", f"accuracy: {run['metrics']['accuracy']:.4f}"]
        macro = run["metrics"]["macro avg"]
        assert f"  macro avg: precision {macro['precision']:.4f}, recall {macro['recall']:.4f}, f1-score {macro['f1-score']:.4f}, support 6" in lines
        # The classes have their names
        assert any(line.startswith(f"  {names[run['predictions']['real_labels'][0]]}: precision ") for line in lines)
        # What was expected, by its name, and what the model predicts
        rows = list(csv.reader(output.open(encoding="utf-8", newline="")))
        assert rows[0][-2:] == ["label", "prediction"]
        assert [row[-2] for row in rows[1:]] == [names[index] for index in run["predictions"]["real_labels"]]
        assert [row[-1] for row in rows[1:]] == [names[index] for index in run["predictions"]["system_predictions"]]

    # Another split of the saved dataset, and one that it does not have
    completed = run_predict(
        trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--evaluate", synthetic_data / "cas" / "local_hf_cls", "--split", "train")
    assert completed.stdout.splitlines()[0].startswith("12 examples of ")
    completed = run_predict(
        trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--input", synthetic_data / "cas" / "local_hf_cls", "--split", "dev")
    assert completed.returncode == 1 and "has no split dev: its splits are train, validation, test. Give one with --split." in completed.stderr

    # A class that the model does not have, and a corpus without any example
    write_table(annotated, [["tokens", "label"], ["une toux", "unknown"]])
    completed = run_predict(trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--evaluate", annotated)
    assert completed.returncode == 1 and completed.stderr.splitlines()[-1].startswith("Example 1: Unknown class 'unknown': the classes are ")
    write_table(annotated, [["tokens", "label"]])
    completed = run_predict(trained_repository, "--task", "cas-cls", "--model", tiny_bert, "--evaluate", annotated)
    assert (completed.returncode, completed.stderr.splitlines()[-1]) == (1, f"{annotated} has no example")


@pytest.mark.e2e
def test_directory_of_a_kept_model_is_enough(trained_repository, tiny_bert, tmp_path):
    """A kept model that was moved: what the run wrote next to it says what it is."""
    run = runs_of(trained_repository, "cas-cls")[0]
    moved = tmp_path / "my-model"
    shutil.copytree(trained_repository / "recipes" / "cas" / "scripts" / run["model_name"], moved)

    completed = run_predict(trained_repository, "--model-dir", moved, "le patient présente une douleur", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() in synthetic.NEGATION_CLASSES
    assert f"Model: tiny-bert fine-tuned on cas-cls, seed 0 (validation accuracy: {run['validation']['score']:.4f})" in completed.stderr

    # A model that an earlier version of the scripts kept has no description: its task has to be given
    (moved / "drbenchmark.json").unlink()
    completed = run_predict(trained_repository, "--model-dir", moved, "le patient présente une douleur", cwd=tmp_path)
    assert completed.returncode == 1
    assert "has no drbenchmark.json, which says what the task of the model is" in completed.stderr and "Give its task with --task." in completed.stderr
    completed = run_predict(trained_repository, "--model-dir", moved, "--task", "cas-cls", "le patient présente une douleur", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() in synthetic.NEGATION_CLASSES


# ---------------------------------------------------------------------------------------------
# The notebook
# ---------------------------------------------------------------------------------------------

NOTEBOOK_RUNNER = """
import json, sys, types

repository, task, model = sys.argv[1:4]
sys.path.insert(0, repository)
shown = []
display = types.ModuleType("IPython.display")
display.HTML = lambda html: html
display.display = shown.append
sys.modules["IPython"] = types.ModuleType("IPython")
sys.modules["IPython.display"] = display

typed = iter(["le patient présente une douleur", "pas de fièvre", ""])
names = {"input": lambda prompt="": next(typed)}
for cell in json.load(open(repository + "/predict_demo.ipynb", encoding="utf-8"))["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        source = source.replace('TASK = "cas-pos"', "TASK = " + repr(task)).replace('MODEL = "Dr-BERT/DrBERT-7GB"', "MODEL = " + repr(model))
        source = source.replace("Le patient présente une toux sèche depuis trois jours, sans fièvre.", "une douleur thoracique")
        exec(compile(source, "cell", "exec"), names)
print(json.dumps(shown))
"""


@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "cas-cls"])
def test_notebook_shows_the_prediction_for_what_is_typed(task_name, trained_repository, tiny_bert, tmp_path):
    """The code of the notebook, without Jupyter: one sentence of its own, then the two that are typed."""
    completed = subprocess.run(
        [sys.executable, "-c", NOTEBOOK_RUNNER, str(trained_repository), task_name, str(tiny_bert)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    lines = completed.stdout.splitlines()
    # The list of the runs, then the model that is loaded
    assert lines[0].split() == ["task", "model", "seed", "validation", "test", "kept", "model"]
    assert f"Model: tiny-bert fine-tuned on {task_name}, seed " in completed.stdout
    assert "The task reads: tokens" in lines

    shown = json.loads(lines[-1])
    assert len(shown) == 3
    if task_name == "cas-pos":
        # Every word has a part of speech: it is highlighted
        assert [re.findall(r">([^<>]+) <small", html) for html in shown] == [
            ["une", "douleur", "thoracique"], ["le", "patient", "présente", "une", "douleur"], ["pas", "de", "fièvre"]]
    else:
        assert all(re.fullmatch(r"<div><b>(\w+)</b></div>", html).group(1) in synthetic.NEGATION_CLASSES for html in shown)


def test_notebook_is_one(predict):
    """The notebook is a file that Jupyter opens, without the outputs of someone's models."""
    notebook = read_json(REPO / "predict_demo.ipynb")
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    cells = notebook["cells"]
    assert len({cell["id"] for cell in cells}) == len(cells)
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)
    # It uses the functions of `predict.py`
    used = set(re.findall(r"predict\.(\w+)", "".join("".join(cell["source"]) for cell in code)))
    assert used and all(hasattr(predict, name) for name in used)
    for name in re.findall(r"model\.(\w+)", "".join("".join(cell["source"]) for cell in code)):
        assert name == "task" or hasattr(predict.Model, name)


# ---------------------------------------------------------------------------------------------
# The models that are kept after the runs: tools/RemoveKeptModels.py
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def kept_models(tmp_path):
    """A copy of the layout of the repository with the tools, and runs that kept their models."""
    repository = tmp_path / "repository"
    repository.mkdir()
    copy_tools(repository)
    runs = Runs(repository / "recipes")
    paths = {
        "first": runs.write(seed=0, validation=0.5),
        "best": runs.write(seed=1, validation=0.75),
        "few-shot": runs.write(seed=2, validation=0.875, fewshot=0.25),
        "not kept": runs.write(seed=3, validation=0.9, kept=None),
        "other task": runs.write(task="cas-cls", seed=0),
        "other model": runs.write(seed=0, model="another/model"),
    }
    for path in paths.values():
        if runs.model_dir(path).is_dir():
            (runs.model_dir(path) / "model.safetensors").write_bytes(b"0" * 1500)
    return repository, runs, paths


def models_there(runs, paths):
    return sorted(name for name, path in paths.items() if runs.model_dir(path).is_dir())


def test_models_that_predict_py_does_not_take_are_listed(kept_models, tmp_path):
    repository, runs, paths = kept_models
    completed = run_tool("RemoveKeptModels.py", repository, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines() == [
        f"{runs.model_dir(paths['first'])}  1.5 kB  (cas-pos, {MODEL}, seed 0)",
        f"{runs.model_dir(paths['few-shot'])}  1.5 kB  (cas-pos, {MODEL}, seed 2, 0.25 of the training set)",
        "2 models to remove, 3.0 kB: 3 models stay. Nothing was removed: give --remove to remove them.",
    ]
    # Nothing is removed without being asked
    assert models_there(runs, paths) == ["best", "few-shot", "first", "other model", "other task"]


def test_models_that_predict_py_does_not_take_are_removed(kept_models, tmp_path, predict, monkeypatch):
    repository, runs, paths = kept_models
    completed = run_tool("RemoveKeptModels.py", repository, "--remove", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines()[-1] == "2 models removed, 3.0 kB: 3 models stay."
    assert models_there(runs, paths) == ["best", "other model", "other task"]
    # The results of the runs are left
    assert all(path.is_file() for path in paths.values())

    # The model that `predict.py` takes is the same one, and nothing else is to remove
    monkeypatch.setattr(predict, "RECIPES_DIR", str(repository / "recipes"))
    assert predict.chosen_run("cas-pos", MODEL)[0].path == str(paths["best"])
    again = run_tool("RemoveKeptModels.py", repository, "--remove", cwd=tmp_path)
    assert again.stdout.splitlines() == ["No model to remove: 3 models stay."]


def test_only_the_directories_that_the_runs_save_are_removed(kept_models, tmp_path):
    """Results that give another directory for their model, e.g. the model that was fine-tuned: it is left."""
    repository, runs, paths = kept_models
    other = repository / "models" / "some-org_some-model"
    other.mkdir(parents=True)
    (other / "config.json").write_text("{}")
    results = read_json(paths["first"])
    results["model_name"] = "../../../models/some-org_some-model"
    paths["first"].write_text(json.dumps(results))

    completed = run_tool("RemoveKeptModels.py", repository, "--all", "--remove", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines()[-1] == "4 models removed, 6.0 kB: 0 models stay."
    assert (other / "config.json").is_file()
    # The model that the run really saved was not named by its results any more
    assert models_there(runs, paths) == ["first"]


def test_every_model_of_a_task_is_removed_when_it_is_asked(kept_models, tmp_path):
    repository, runs, paths = kept_models
    completed = run_tool("RemoveKeptModels.py", repository, "--task", "cas-pos", "--model", MODEL, "--all", "--remove", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.splitlines()[-1] == "3 models removed, 4.5 kB: 0 models stay."
    assert models_there(runs, paths) == ["other model", "other task"]

    completed = run_tool("RemoveKeptModels.py", repository, "--task", "cas-cls", cwd=tmp_path)
    assert completed.stdout.splitlines() == ["No model to remove: 1 model stays."]
