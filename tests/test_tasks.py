"""Tests of what the script of a recipe states: its task.

A script creates its task in a line `TASK = ...`, with what is specific to it. What a
model reads for an example is built by the task, in one place: these tests give examples
to the tasks, without any training.
"""

import sys

import pytest

import synthetic
from helpers import ALL_TASKS, Task, load_module


def task_ids(task):
    return task.name


def task_of(name):
    """The task that the script of a task of `run.py` creates."""
    task = Task(name)
    return load_module("drbenchmark_script_of_" + name.replace("-", "_"), task.scripts_dir / task.script).TASK


@pytest.fixture(scope="module")
def tokenizer(tiny_drbert):
    """A tokenizer whose special tokens are the ones that the scripts write in their texts: <s> and </s>."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_drbert)


@pytest.fixture(scope="module")
def bert_tokenizer(tiny_bert):
    """A tokenizer that splits three words into two sub-tokens."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_bert)


@pytest.fixture(scope="module")
def token_classification():
    """The module of the tasks that label words, as their scripts import it."""
    return sys.modules[type(task_of("cas-pos")).__module__]


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_training_script_defines_its_task(task):
    """A script states its task in one line, and only trains when it is started."""
    lines = (task.scripts_dir / task.script).read_text().splitlines()
    assert sum(line.startswith("TASK = ") for line in lines) == 1
    assert lines[-2:] == ["if __name__ == '__main__':", "    TASK.train()"]


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_task_names_the_columns_that_its_model_reads(task):
    """They are columns of the dataset of the task: the words of a sentence, or a text."""
    defined = task_of(task.name)
    assert isinstance(defined.kind, str) and defined.kind
    assert hasattr(defined.model_class, "from_pretrained")

    example = synthetic.dataset_of(task.recipe, task.subset)["train"][0]
    assert defined.inputs
    for column, kind in defined.inputs.items():
        value = example[column]
        if kind == "words":
            assert isinstance(value, list) and all(isinstance(word, str) for word in value), column
        else:
            assert kind == "text" and isinstance(value, str), column


QUESTION = {"question": "q", "answer_a": "a", "answer_b": "b", "answer_c": "c", "answer_d": "d", "answer_e": "e"}


@pytest.mark.parametrize("task_name, example, text", [
    # The words of a sentence are joined
    ("cas-cls", {"tokens": ["pas", "de", "toux"]}, "pas de toux"),
    ("pxcorpus-cls", {"tokens": ["pas", "de", "toux"]}, "pas de toux"),
    # The stop words are removed from the text, which is put in lower case
    ("diamed-cls", {"clinical_case": "Le patient présente une toux , sans fièvre"}, "patient présente toux fièvre"),
    # Several texts are joined with the special tokens of the model
    ("deft2020-cls", {"source": "s", "cible_1": "a", "cible_2": "b", "cible_3": "c"}, "<s> s </s> (1) a </s> (2) b </s> (3) c </s>"),
    ("frenchmedmcqa-cls", QUESTION, "<s> q </s> a </s> b </s> c </s> d </s> e </s>"),
    ("frenchmedmcqa-mcqa", QUESTION, "<s> q </s> a </s> b </s> c </s> d </s> e </s>"),
    ("clister-regr", {"text_1": "une toux", "text_2": "une fièvre"}, "<s> une toux </s>  une fièvre </s>"),
    ("deft2020-regr", {"source": "une toux", "cible": "une fièvre"}, "<s> une toux </s>  une fièvre </s>"),
])
def test_text_that_a_model_reads(task_name, example, text, tokenizer):
    """Each task builds its text in its own way: `encode` is where it is done."""
    encoded = task_of(task_name).encode(example, tokenizer, 512, "some/model")
    assert encoded["text"] == text
    assert encoded["inputs"] == dict(tokenizer(text, truncation=True, max_length=512, padding="do_not_pad"))


@pytest.mark.parametrize("task_name, column", [("morfitt-cls", "abstract"), ("deft2021-cls", "text")])
def test_text_of_a_dataset_is_read_as_it_is(task_name, column, tokenizer):
    encoded = task_of(task_name).encode({column: "une toux sans fièvre"}, tokenizer, 512, "some/model")
    assert encoded == {"inputs": dict(tokenizer("une toux sans fièvre", truncation=True, max_length=512, padding="do_not_pad"))}


@pytest.mark.parametrize("task_name, example", [
    ("cas-cls", {"tokens": ["le", "patient", "présente", "une", "toux"]}),
    ("morfitt-cls", {"abstract": "le patient présente une toux"}),
    ("clister-regr", {"text_1": "le patient présente", "text_2": "une toux"}),
])
def test_text_is_cut_at_the_length_that_is_given(task_name, example, tokenizer):
    task = task_of(task_name)
    whole = task.encode(example, tokenizer, 512, "some/model")["inputs"]["input_ids"]
    cut = task.encode(example, tokenizer, 4, "some/model")["inputs"]["input_ids"]
    # The last token is the one that the tokenizer adds
    assert len(whole) > 4 and cut == whole[:3] + whole[-1:]


def test_words_that_a_model_reads(bert_tokenizer, tiny_bert):
    """The words of a sentence are given to the tokenizer as they are: it says which word each sub-token comes from."""
    task = task_of("quaero-ner_emea")
    words = ["douleur", "thoracique", "aiguë"]
    encoded = task.encode({"tokens": words}, bert_tokenizer, 512, str(tiny_bert))
    assert encoded["inputs"] == {name: values[0] for name, values in bert_tokenizer([words], is_split_into_words=True).items()}
    assert (encoded["word_ids"], encoded["words"]) == ([None, 0, 1, 1, 2, None], 3)
    # A sentence that is too long loses its last words
    assert task.encode({"tokens": words}, bert_tokenizer, 4, str(tiny_bert))["word_ids"] == [None, 0, 1, None]


def test_words_are_tokenized_one_by_one_for_flaubert(token_classification, tiny_flaubert, bert_tokenizer, tiny_bert):
    """The tokenizer of FlauBERT does not say which word a sub-token comes from: the words are given to it one by one."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_flaubert)
    sentences = [["le", "patient"], ["toux"]]
    # The tiny tokenizer splits a word into its letters
    inputs, word_ids = token_classification.tokenize(tokenizer, sentences, 512, "flaubert/flaubert_base_cased")
    assert word_ids == [[None, 0, 0, 1, 1, 1, 1, 1, 1, 1, None], [None, 0, 0, 0, 0, None]]
    assert list(inputs) == ["input_ids"] and [len(ids) for ids in inputs["input_ids"]] == [11, 6]
    first, last = tokenizer("<s>")["input_ids"][1], tokenizer("</s>")["input_ids"][1]
    assert all((ids[0], ids[-1]) == (first, last) for ids in inputs["input_ids"])
    assert inputs["input_ids"][1][1:-1] == tokenizer("toux")["input_ids"][1:-1]

    # A sentence that is too long is cut, and keeps its last token
    cut, cut_word_ids = token_classification.tokenize(tokenizer, sentences, 6, "flaubert/flaubert_base_cased")
    assert cut_word_ids == [[None, 0, 0, 1, 1, None], [None, 0, 0, 0, 0, None]]
    assert cut["input_ids"][0] == inputs["input_ids"][0][:5] + [last]
    # Without a length, the one of the tokenizer
    assert token_classification.tokenize(tokenizer, sentences, None, "flaubert/flaubert_base_cased")[1] == word_ids

    # The other models have a tokenizer that says it: the words of a sentence are given together
    inputs, word_ids = token_classification.tokenize(bert_tokenizer, [["douleur", "thoracique"], ["toux"]], 512, str(tiny_bert))
    assert word_ids == [[None, 0, 1, 1, None], [None, 0, None]]
    assert set(inputs) == {"input_ids", "token_type_ids", "attention_mask"}


def test_labels_of_the_sub_tokens(token_classification):
    word_ids = [None, 0, 0, 1, 2, 2, None]
    # A word gives its label to its first sub-token
    assert token_classification.align_labels([3, 5, 7], word_ids, False) == [-100, 3, -100, 5, 7, -100, -100]
    # With FlauBERT, to all of them
    assert token_classification.align_labels([3, 5, 7], word_ids, True) == [-100, 3, 3, 5, 7, 7, -100]


def test_class_of_a_question_is_the_set_of_its_right_answers():
    """FrenchMedMCQA: a question has several right answers among five, and the model one class for each set of answers."""
    task = task_of("frenchmedmcqa-mcqa")
    features = synthetic.dataset_of("frenchmedmcqa", None)["train"].features
    assert len(task.labels) == len(set(task.labels)) == 31
    # The answers b and a, in the order of the dataset
    assert task.labels[task.target({"correct_answers": [1, 0]}, task.labels, features)] == "ab"
    assert task.labels[task.target({"correct_answers": [4]}, task.labels, features)] == "e"

    # The scores of the task: the part of the answers that are right, and the questions that are entirely right
    assert task.metrics(["ab", "c", "a"], ["ab", "d", "ab"], task.labels) == {"hamming_score": 0.5, "exact_match": 1 / 3}
