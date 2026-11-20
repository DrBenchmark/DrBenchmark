"""Show the examples of the test set that a run got wrong, and the classes that it confused.

The results of a run (`recipes/<recipe>/runs/*.json`) have its predictions on the test set,
but not the text of the examples, and the classes of some tasks are numbers there. The test
set is read again to show, for one run:

- the confusions: how many times a class was predicted in place of another one;
- the examples with an error: their text, what was expected and what was predicted.

The run is the one of a task for a model, on the whole training set (`--task`, `--model`, and
`--seed` when there are several runs), or the one of a results file (`--run`).
"""

import os
import sys
import json
import argparse
from glob import glob
from collections import Counter

from models_and_datasets import label_names, load_benchmark_dataset
from repository_paths import RECIPES_DIR

# The kinds of predictions that the recipes write, for each example of the test set:
# one label per word (part-of-speech tagging, named-entity recognition)
TOKENS = "tokens"
# one class (classification)
CLASSES = "classes"
# 0 or 1 for each class (multi-label classification)
LABEL_SETS = "label sets"
# the letters of the right answers (multiple-choice question answering)
ANSWERS = "answers"
# a number (semantic textual similarity)
SCORES = "scores"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, required=False, default=None,
                        help="Task of the run: one of the names of `python run.py --help`.")
    parser.add_argument("--model", type=str, required=False, default=None,
                        help="Model of the run: its name on the HuggingFace Hub, or its path.")
    parser.add_argument("--seed", type=int, required=False, default=None,
                        help="Seed of the run (default: the lowest one of the runs of the task for the model).")
    parser.add_argument("--run", type=str, required=False, default=None, metavar="PATH",
                        help="Results file of the run, instead of --task and --model.")
    parser.add_argument("--offline", action="store_true",
                        help="Read the test set where download_datasets_locally.py saves it, instead of the HuggingFace Hub.")
    parser.add_argument("--max-confusions", type=int, required=False, default=20, metavar="N",
                        help="Number of confusions to show, the most frequent ones first (default: 20). All of them with 0.")
    parser.add_argument("--max-examples", type=int, required=False, default=20, metavar="N",
                        help="Number of examples with an error to show (default: 20). All of them with 0.")
    parser.add_argument("--max-length", type=int, required=False, default=300, metavar="N",
                        help="Number of characters of each text that are shown (default: 300). All of them with 0.")
    args = parser.parse_args()
    by_names, by_file = args.task is not None and args.model is not None, args.run is not None
    if by_names == by_file or (by_file and (args.task is not None or args.model is not None)):
        parser.error("give the run either by --task and --model, or by --run")
    return args


def read_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_model(hyperparameters, model):
    """Whether a run is the one of a model, given by its name on the HuggingFace Hub or by its path.

    A run in offline mode has the directory of the copy of its model, `models/<organisation>_<name>`
    in lower case, and its name on the Hub when it was computed by the recent scripts.
    """
    path = str(hyperparameters.get("model_name") or "")
    local_copy = os.path.basename(path.rstrip("/"))
    return model in (hyperparameters.get("hub_model_name"), path) or local_copy == model.lower().replace("/", "_")


def chosen_run(args):
    """Path and results of the run that is asked for, and the seeds of the other runs it was chosen among."""
    if args.run is not None:
        return args.run, read_results(args.run), []

    directory = os.path.join(RECIPES_DIR, args.task.split("-")[0], "runs")
    # The runs of the task on the whole training set
    runs = [(path, read_results(path)) for path in sorted(glob(os.path.join(directory, "*.json")))]
    runs = [
        (path, results) for path, results in runs
        if results["hyperparameters"].get("task") == args.task and results["hyperparameters"].get("fewshot", 1.0) == 1.0
    ]
    of_the_model = sorted(
        ((path, results) for path, results in runs if is_model(results["hyperparameters"], args.model)),
        key=lambda run: run[1]["hyperparameters"]["seed"],
    )
    if not of_the_model:
        models = sorted({str(results["hyperparameters"].get("hub_model_name") or results["hyperparameters"]["model_name"])
                         for _, results in runs})
        sys.exit(f"No run of {args.task} for {args.model} in {directory}.\n"
                 + ("The models that have runs of this task: " + ", ".join(models) if models else "No model has a run of this task."))

    seeds = [results["hyperparameters"]["seed"] for _, results in of_the_model]
    if args.seed is None:
        path, results = of_the_model[0]
    elif args.seed in seeds:
        path, results = of_the_model[seeds.index(args.seed)]
    else:
        sys.exit(f"No run of {args.task} for {args.model} has the seed {args.seed}: the seeds of its runs are "
                 + ", ".join(str(seed) for seed in seeds) + ".")
    return path, results, [seed for seed in seeds if seed != results["hyperparameters"]["seed"]]


def kind_of(references):
    """Kind of the predictions of a run, from what it wrote as the references of the test set."""
    first = references[0]
    if isinstance(first, list):
        labels = [label for reference in references for label in reference]
        return TOKENS if all(isinstance(label, str) for label in labels) else LABEL_SETS
    if isinstance(first, str):
        return ANSWERS
    return CLASSES if isinstance(first, int) else SCORES


def examples_of(test, identifiers, number):
    """The examples of the test set that the run has predictions for, and the column of their identifier.

    A run predicts the examples of the test set in their order: all of them, or the first
    ones with `--max_test_samples`. The test set that is read has to be the one of the run.
    """
    if len(test) < number:
        sys.exit(f"The test set has {len(test)} examples and the run has {number} predictions: it is not the test set of the run.")
    examples = [test[index] for index in range(number)]
    id_column = None
    for column in test.column_names:
        if [example[column] for example in examples] == list(identifiers[:number]):
            id_column = column
            break
    if id_column is None and "id" in test.column_names:
        sys.exit("The identifiers of the test set are not the ones of the run: it is not the test set of the run.")
    return examples, id_column


def names_of_classes(test, kind, references):
    """Names of the classes that a run wrote as numbers, or None when the test set does not give them.

    They are the ones of the column of the test set that has the references of the run.
    """
    for column, feature in test.features.items():
        try:
            names = label_names(feature)
        except AttributeError:
            # A column of texts or of numbers
            continue
        if kind == CLASSES and hasattr(feature, "names") and list(test[column])[:len(references)] == references:
            return names
        if kind == LABEL_SETS and not hasattr(feature, "names") and len(names) == len(references[0]):
            return names
    return None


def texts_of(example, id_column, max_length):
    """The texts of an example: [(column, text)]. A list of words is shown as a text."""
    texts = []
    for column, value in example.items():
        if column == id_column:
            continue
        if isinstance(value, list) and value and all(isinstance(word, str) for word in value):
            value = " ".join(value)
        if isinstance(value, str) and value:
            if max_length and len(value) > max_length:
                value = value[:max_length] + "..."
            texts.append((column, value))
    return texts


def labels_of(vector, names):
    """The classes of an example of a multi-label task, which are written as 0 or 1 for each class."""
    return [names[index] if names else str(index) for index, value in enumerate(vector) if value >= 0.5]


def errors_of(kind, references, predictions, names):
    """For each example, what was expected and what was predicted, as texts, and its confusions.

    The confusions of an example are the pairs (expected, predicted) that differ: one per word,
    one per example, or one per class that is missing or added for a multi-label task. A
    similarity score has no class: its error is the distance to the expected score.
    """
    def name(index):
        return names[index] if names else str(index)

    described = []
    for reference, prediction in zip(references, predictions):
        if kind == TOKENS:
            confusions = [(expected, predicted) for expected, predicted in zip(reference, prediction) if expected != predicted]
            described.append((None, None, confusions))
        elif kind == CLASSES:
            expected, predicted = name(reference), name(prediction)
            described.append((expected, predicted, [(expected, predicted)] if reference != prediction else []))
        elif kind == LABEL_SETS:
            expected, predicted = labels_of(reference, names), labels_of(prediction, names)
            confusions = [(label, "(missing)") for label in expected if label not in predicted]
            confusions += [("(none)", label) for label in predicted if label not in expected]
            described.append((", ".join(expected) or "(none)", ", ".join(predicted) or "(none)", confusions))
        elif kind == ANSWERS:
            described.append((reference, prediction, [(reference, prediction)] if reference != prediction else []))
        else:
            described.append((f"{reference:.2f}", f"{prediction:.2f}", abs(reference - prediction)))
    return described


def words_with_errors(words, reference, prediction):
    """The words of a sentence, each wrong one followed by [expected -> predicted]."""
    shown = []
    for index, (expected, predicted) in enumerate(zip(reference, prediction)):
        word = words[index] if index < len(words) else "?"
        shown.append(word if expected == predicted else f"{word} [{expected} -> {predicted}]")
    return " ".join(shown)


def main():
    args = arguments()
    path, results, other_seeds = chosen_run(args)
    hyperparameters, predictions = results["hyperparameters"], results["predictions"]
    references, predicted = predictions["real_labels"], predictions["system_predictions"]
    if not references:
        sys.exit(f"{path} has no prediction.")

    test = load_benchmark_dataset(hyperparameters["dataset"], hyperparameters["subset"], args.offline)["test"]
    examples, id_column = examples_of(test, predictions["identifiers"], len(references))
    kind = kind_of(references)
    names = names_of_classes(test, kind, references) if kind in (CLASSES, LABEL_SETS) else None
    described = errors_of(kind, references, predicted, names)

    model = hyperparameters.get("hub_model_name") or hyperparameters["model_name"]
    print(f"Run: {path}")
    print(f"Task {hyperparameters['task']}, model {model}, seed {hyperparameters['seed']}"
          + (f" (other seeds: {', '.join(str(seed) for seed in other_seeds)})" if other_seeds else ""))

    if kind == SCORES:
        distances = [distance for _, _, distance in described]
        # Every example has an error: the ones that are the farthest from the expected score are shown
        wrong = sorted(range(len(described)), key=lambda index: -distances[index])
        print(f"{len(described)} examples of the test set: the predicted score is at "
              f"{sum(distances) / len(distances):.2f} of the expected one on average, and at {max(distances):.2f} at most")
    else:
        wrong = [index for index, (_, _, confusions) in enumerate(described) if confusions]
        print(f"{len(described)} examples of the test set, {len(wrong)} with an error")

        confusions = Counter(confusion for _, _, confusions in described for confusion in confusions)
        if kind == TOKENS:
            print(f"\nConfusions: {sum(confusions.values())} wrong labels among the ones of {sum(len(reference) for reference in references)} words")
        else:
            print(f"\nConfusions: {sum(confusions.values())}")
        print("    expected -> predicted: number of times")
        frequent = sorted(confusions.items(), key=lambda item: (-item[1], item[0]))
        for (expected, predicted_class), count in frequent[:args.max_confusions or None]:
            print(f"    {expected} -> {predicted_class}: {count}")
        if args.max_confusions and len(frequent) > args.max_confusions:
            others = frequent[args.max_confusions:]
            print(f"    ... and {len(others)} other confusions, {sum(count for _, count in others)} times in all")

    shown = wrong[:args.max_examples] if args.max_examples else wrong
    print(f"\nExamples with {'the largest error' if kind == SCORES else 'an error'}: {len(shown)} of {len(wrong)}")
    for index in shown:
        example = examples[index]
        expected, predicted_class, confusions = described[index]
        identifier = example[id_column] if id_column is not None else predictions["identifiers"][index]
        if kind == TOKENS:
            print(f"\n[{identifier}] {len(confusions)} wrong label{'s' if len(confusions) > 1 else ''}")
            print("    " + words_with_errors(example.get("tokens") or [], references[index], predicted[index]))
            continue
        distance = f" (distance: {confusions:.2f})" if kind == SCORES else ""
        print(f"\n[{identifier}] expected: {expected} | predicted: {predicted_class}{distance}")
        for column, text in texts_of(example, id_column, args.max_length):
            print(f"    {column}: {text}")


if __name__ == '__main__':
    main()
