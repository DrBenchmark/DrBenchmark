"""Tests of the tokenizer of the DrBERT models, the default models of the recipes.

These models name the tokenizer class `CamembertTokenizer`, which is a Unigram model, while
the tokenizer they store in `tokenizer.json` is a BPE model. transformers 4 reads
`tokenizer.json` as it is, whatever the class. `AutoTokenizer` of transformers 5 builds the
Unigram model of the class and fails ("too many values to unpack"): this is why
`requirements.txt` asks for `transformers<5`.

`tiny_drbert` is a model with the same tokenizer files: these tests have to pass before the
bound of transformers is raised.
"""

import json

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

import synthetic

SENTENCES = [synthetic.text(i) for i in range(len(synthetic.WORDS))]


def load_tokenizer(model):
    """The tokenizer of a model, loaded as the training scripts load it."""
    return AutoTokenizer.from_pretrained(str(model), use_fast=True)


def test_drbert_tokenizer_is_the_one_of_tokenizer_json(tiny_drbert):
    tokenizer = load_tokenizer(tiny_drbert)
    stored = Tokenizer.from_file(str(tiny_drbert / "tokenizer.json"))

    assert tokenizer.tokenize("douleur thoracique aiguë") == ["▁douleur", "▁thor", "a", "c", "i", "q", "u", "e", "▁aiguë"]
    for sentence in SENTENCES:
        assert tokenizer(sentence)["input_ids"] == stored.encode(sentence).ids
    first, second = SENTENCES[:2]
    assert tokenizer(first, second)["input_ids"] == stored.encode(first, second).ids

    special_tokens = (tokenizer.cls_token, tokenizer.sep_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token, tokenizer.mask_token)
    assert special_tokens == ("<s>", "</s>", "</s>", "<pad>", "<unk>", "<mask>")
    # The regression and MCQA scripts write the special tokens in the text they give to the tokenizer
    text = f"{tokenizer.cls_token} {first} {tokenizer.sep_token}  {second} {tokenizer.eos_token}"
    identifiers = tokenizer(text)["input_ids"]
    assert identifiers == stored.encode(text).ids
    assert identifiers.count(tokenizer.cls_token_id) == 2 and identifiers.count(tokenizer.sep_token_id) == 3

    # Every identifier has an embedding in the model
    assert len(tokenizer) == json.loads((tiny_drbert / "config.json").read_text())["vocab_size"]


def test_drbert_tokenizer_aligns_sub_tokens_and_words(tiny_drbert):
    """The token classification script needs to know the word of each sub-token."""
    encoding = load_tokenizer(tiny_drbert)([["douleur", "thoracique", "aiguë"]], is_split_into_words=True)
    assert encoding.word_ids(batch_index=0) == [None, 0] + [1] * 7 + [2, None]


# One task for each training script that loads a tokenizer: the four kinds of tasks of
# `recipes/common/`, and the MCQA script, which also reads the tokenizer saved with the best model
@pytest.mark.e2e
@pytest.mark.parametrize("task_name", ["cas-pos", "cas-cls", "morfitt-cls", "clister-regr", "frenchmedmcqa-mcqa"])
def test_recipe_with_the_tokenizer_of_drbert(task_name, run_recipe, tiny_drbert):
    """The task is run as `run.py` runs it, with a model that has the tokenizer files of DrBERT."""
    predictions = run_recipe(task_name, tiny_drbert).result["predictions"]
    assert len(predictions["system_predictions"]) == synthetic.SPLIT_SIZES["test"]
    if task_name == "cas-pos":
        # One predicted label per word: the sub-tokens were aligned with the words
        for reference, prediction in zip(predictions["real_labels"], predictions["system_predictions"]):
            assert len(reference) == len(prediction)
