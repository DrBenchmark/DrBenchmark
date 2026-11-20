#!/usr/bin/env python3

# Copyright  2023 Yanis Labrak (Avignon University - LIA)
#            2023 Mickael Rouvier (Avignon University - LIA)
# Apache 2.0

"""Fine-tuning for the token classification tasks: part-of-speech tagging and named-entity recognition."""

import logging
import os

import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification

from . import training, utils

# The label of a word that is not in an entity. It is also the one of the words that a model
# does not read: the ones that are cut from a sentence which is too long.
OUTSIDE = "O"


def word_by_word(model_name):
    """Whether the words of a sentence are tokenized one by one.

    The tokenizer of FlauBERT does not say which word a sub-token comes from.
    """
    return "flaubert" in model_name.lower()


def tokenize(tokenizer, sentences, max_length, model_name):
    """Tokenize sentences that are lists of words, for the model named `model_name`.

    Returns what is given to the model for the sentences, and for each sentence the word of
    each of its sub-tokens: its number in the sentence, or None for a token that the tokenizer adds.
    """
    if word_by_word(model_name):
        max_length = max_length or tokenizer.model_max_length
        input_ids, word_ids = [], []

        for words in sentences:

            _local = [tokenizer("<s>")["input_ids"][1]]
            _local_words = [None]

            for index, word in enumerate(words):
                tokens_word = tokenizer(word)["input_ids"][1:-1]
                _local.extend(tokens_word)
                _local_words.extend([index] * len(tokens_word))

            if len(_local) > 250:
                logging.info(f">> {len(_local)}")

            _local = _local[0:max_length - 1]
            _local_words = _local_words[0:max_length - 1]

            _local.append(tokenizer("</s>")["input_ids"][1])
            _local_words.append(None)

            input_ids.append(_local)
            word_ids.append(_local_words)

        return {"input_ids": input_ids}, word_ids

    tokenized_inputs = tokenizer(list(sentences), truncation=True, max_length=max_length, padding="do_not_pad", is_split_into_words=True)
    return tokenized_inputs, [tokenized_inputs.word_ids(batch_index=i) for i in range(len(tokenized_inputs["input_ids"]))]


def align_labels(tags, word_ids, label_all_tokens):
    """Labels of the sub-tokens of a sentence, from the tags of its words: -100 where the model does not learn one.

    A word gives its tag to its first sub-token, or to all of them with `label_all_tokens`.
    """
    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            label_ids.append(tags[word_idx])

        else:
            label_ids.append(tags[word_idx] if label_all_tokens else -100)

        previous_word_idx = word_idx

    return label_ids


def load_seqeval(experiment_id):
    """The metric of the tasks: precision, recall and F1 of the entities, and accuracy of the labels."""
    return evaluate.load(os.path.join(utils.REPO_DIR, "metrics", "seqeval.py"), experiment_id=experiment_id)


class Task(training.Task):
    """A task that gives a label to each word of a sentence.

    Args:
        dataset: name of the dataset on the HuggingFace Hub, or path of its loading script.
        tag_column: column with the tags of the tokens.
        metric_for_best_model: metric computed on the validation set to choose the best
            checkpoint: "f1", "accuracy", "precision", "recall" or "macro_f1".
        needs_data_dir: give `--data_dir` to the loading script of the dataset.
    """

    kind = "token classification"
    model_class = AutoModelForTokenClassification
    inputs = {"tokens": "words"}
    word_labels = True

    def __init__(self, dataset, tag_column="ner_tags", metric_for_best_model="f1", needs_data_dir=False):
        self.dataset = dataset
        self.tag_column = tag_column
        self.reference_column = tag_column
        self.metric_for_best_model = metric_for_best_model
        self.needs_data_dir = needs_data_dir

    def encode(self, example, tokenizer, max_length, model_name):
        words = list(example["tokens"])
        inputs, word_ids = tokenize(tokenizer, [words], max_length, model_name)
        return {"inputs": {name: values[0] for name, values in inputs.items()}, "word_ids": word_ids[0], "words": len(words)}

    def decode(self, logits, encoded, labels):
        """One label per word: the one of its first sub-token, and `O` for a word that the model did not read."""
        predicted = np.argmax(logits, axis=-1)
        prediction = [OUTSIDE] * encoded["words"]
        seen = set()
        for position, word in enumerate(encoded["word_ids"]):
            if word is not None and word not in seen:
                seen.add(word)
                prediction[word] = labels[predicted[position]]
        return prediction

    def reference(self, example, labels, features=None):
        names = training.class_names(features, self.tag_column) or labels
        return [training.name_of(tag, names) for tag in example[self.tag_column]]

    def metrics(self, references, predictions, labels, identifiers=None):
        metric = load_seqeval(f"metrics-{os.getpid()}")
        return metric.compute(predictions=predictions, references=references, zero_division=.0)

    def train(self):
        tag_column = self.tag_column
        run = training.start(self.dataset, needs_data_dir=self.needs_data_dir, task=self)
        args, dataset = run.args, run.dataset
        model_name = training.model_of(args)

        try:
            label_list = dataset["train"].features[tag_column].feature.names
        except AttributeError:
            # For datasets<4
            label_list = dataset["train"].features[tag_column][0].names
        run.labels = label_list

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
        training.name_outputs(model, label_list)

        # Every sub-token of a word learns its label when the words are tokenized one by one
        label_all_tokens = word_by_word(model_name)
        if label_all_tokens and not args.max_position_embeddings:
            args.max_position_embeddings = tokenizer.model_max_length

        def tokenize_and_align_labels(examples):
            tokenized_inputs, word_ids = tokenize(tokenizer, examples["tokens"], args.max_position_embeddings, model_name)
            tokenized_inputs["labels"] = [
                align_labels(tags, sentence_word_ids, label_all_tokens)
                for tags, sentence_word_ids in zip(examples[tag_column], word_ids)
            ]
            return tokenized_inputs

        train_tokenized_datasets, validation_tokenized_datasets, test_tokenized_datasets = training.prepare_splits(
            run, tokenize_and_align_labels, batched=True, keep_in_memory=True,
        )
        if "label" in dataset["train"].column_names:
            # A column of labels for another task (PxCorpus) would be taken for the tags of the tokens
            train_tokenized_datasets = train_tokenized_datasets.remove_columns(["label"])
            validation_tokenized_datasets = validation_tokenized_datasets.remove_columns(["label"])
            test_tokenized_datasets = test_tokenized_datasets.remove_columns(["label"])

        training_args = training.training_arguments(run, self.metric_for_best_model, greater_is_better=True)

        logging.info('Load Metrics')
        metric = load_seqeval(run.name)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        def compute_metrics(p):
            # Beware : The labels here might be truncated according to `model_max_length`
            # These are the training metrics, not the final metrics.

            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
            true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

            results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=.0)

            macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
            macro_f1 = sum(macro_values) / len(macro_values) if macro_values else 0.0

            return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

        trainer = training.create_trainer(
            model,
            training_args,
            train_tokenized_datasets,
            validation_tokenized_datasets,
            tokenizer,
            compute_metrics,
            data_collator=data_collator,
        )

        training.train(run, trainer)

        logging.info("***** Starting Evaluation *****")

        predictions, labels, _ = trainer.predict(test_tokenized_datasets)
        predictions = np.argmax(predictions, axis=2)

        # Convert predictions at the subtoken level to token level
        _true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # We might have truncated the labels in `tokenize_and_align_labels`
        # We retrieve the untruncated labels
        _true_labels = [[label_list[label] for label in doc] for doc in test_tokenized_datasets[tag_column]]
        # We pad the predictions with 'O' to match untruncated labels
        _true_predictions = [p + [OUTSIDE] * (len(l) - len(p)) for p, l in zip(_true_predictions, _true_labels)]

        cr_metric = metric.compute(predictions=_true_predictions, references=_true_labels, zero_division=.0)
        logging.info(cr_metric)

        training.save_results(
            run,
            trainer,
            metrics=cr_metric,
            identifiers=list(dataset["test"]["id"]),
            real_labels=_true_labels,
            system_predictions=_true_predictions,
        )
