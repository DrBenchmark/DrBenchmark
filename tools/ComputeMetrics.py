"""Collect the scores of the runs of every task, and average them.

The runs are read in `recipes/<recipe>/runs/`. The scores are written in
`stats/results.json`, and their averages in `stats/overall_averaged_metrics.json`.

The runs of a task are averaged for each model: they have to be trained in the same
way. When they are not, nothing is written and the runs that differ are listed.
"""

import os
import sys
import json
import argparse
from glob import glob
from collections import defaultdict

from paths import RECIPES_DIR, REPO_DIR, STATS_DIR

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x, *args, **kwargs):
        return x

# Hyper-parameters that the runs of a task do not have to share to be averaged: the seed is what
# distinguishes them, and the others say where the files are, or name the model and the task
# (the runs are averaged by model, task and proportion of the training set).
NOT_COMPARED = {
    "seed", "config", "model_name", "output_dir", "run_dir", "data_dir", "offline",
    "task", "subset", "dataset", "fewshot",
}
EFFECTIVE_BATCH_SIZE = "effective_batch_size"
UPDATES_PER_EPOCH = "updates_per_epoch"
DESCRIPTIONS = {
    EFFECTIVE_BATCH_SIZE: "batch_size x gradient_accumulation_steps",
    UPDATES_PER_EPOCH: "measured: updates of the weights during the training / epochs",
}


def normalise(value):
    """A value that can be compared between runs written by different versions of the scripts.

    The learning rate, for instance, was written as the string "1e-5" by the first ones.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return json.dumps(value, sort_keys=True)


def configuration(data):
    """What the runs of a task have to share to be averaged: {name: value}."""
    hyperparameters = dict(data["hyperparameters"])

    # 12 examples per batch, or 12 accumulated batches of 1 example, update the weights with the
    # same examples. The runs older than the option did not accumulate.
    batch_size = normalise(hyperparameters.pop("batch_size", None))
    accumulation = normalise(hyperparameters.pop("gradient_accumulation_steps", None)) or 1.0
    config = {name: normalise(value) for name, value in hyperparameters.items() if name not in NOT_COMPARED}
    config[EFFECTIVE_BATCH_SIZE] = batch_size * accumulation if isinstance(batch_size, float) else batch_size

    # What the training did, whatever the hyper-parameters say. Only the recent scripts write it.
    state = data.get("trainer_state") or {}
    if state.get("global_step") and state.get("epoch"):
        config[UPDATES_PER_EPOCH] = float(round(state["global_step"] / state["epoch"]))
    return config


def differences(runs, ignored=()):
    """Hyper-parameters on which runs disagree: {name: {value: [paths of the runs]}}.

    `runs` is a list of (path, configuration).
    """
    names = set().union(*(config for _, config in runs)) - set(ignored)
    different = {}
    for name in sorted(names):
        runs_of_values = defaultdict(list)
        for path, config in runs:
            # Not measured in the old runs: unknown, which is not a difference
            if name == UPDATES_PER_EPOCH and name not in config:
                continue
            # A hyper-parameter that a run does not have did not exist yet: it had no value
            runs_of_values[config.get(name)].append(path)
        values = list(runs_of_values)
        if name == UPDATES_PER_EPOCH:
            # The last batches of an epoch, when they are fewer than the accumulated ones, give
            # one more update or none, depending on the version of transformers
            is_different = max(values) - min(values) > 1
        else:
            is_different = len(values) > 1
        if is_different:
            different[name] = dict(runs_of_values)
    return different


def display(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "not set" if value is None else str(value)


def report(problems):
    """Describe the runs that cannot be averaged: {(model, corpus, task, fewshot): differences}."""
    lines = []
    for (model, corpus, task, fewshot), different in problems.items():
        lines.append(f"{model} | {corpus}-{task} | {fewshot} of the training set")
        for name, runs_of_values in different.items():
            description = f" ({DESCRIPTIONS[name]})" if name in DESCRIPTIONS else ""
            lines.append(f"    {name}{description}")
            for value, paths in sorted(runs_of_values.items(), key=lambda item: -len(item[1])):
                lines.append(f"        {display(value)}: {len(paths)} run(s)")
                lines.extend(f"            {os.path.relpath(path, REPO_DIR)}" for path in sorted(paths))
    names = sorted({name for different in problems.values() for name in different})
    lines += [
        "",
        f"{len(problems)} set(s) of runs cannot be averaged: their runs were not trained in the same way.",
        "Nothing was written. Move the runs that should not be averaged out of the `runs/` directories,",
        f"or accept these differences with: --ignore {' '.join(names)}",
    ]
    return "\n".join(lines)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ignore", type=str, nargs='+', default=[], metavar="NAME",
        help="Hyper-parameters on which the runs of a task may differ. Besides the arguments of the "
             f"training scripts: {EFFECTIVE_BATCH_SIZE} ({DESCRIPTIONS[EFFECTIVE_BATCH_SIZE]}) and "
             f"{UPDATES_PER_EPOCH} ({DESCRIPTIONS[UPDATES_PER_EPOCH]}).")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    path = RECIPES_DIR

    # results: model -> corpus-task -> metric -> list
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # runs that are averaged together: (model, corpus, task, fewshot) -> [(path, configuration)]
    runs = defaultdict(list)

    file_paths = glob(os.path.join(path, '*', 'runs', '*.json'))
    print("Reading run results...")
    for file_path in tqdm(file_paths):

        with open(file_path) as f_json:
            data = json.load(f_json)

        corpus, task = data['hyperparameters']['task'].split('-')

        model_name = data["hyperparameters"]["model_name"]
        if model_name.startswith('../../../models/'):
            model_name = model_name.replace('../../../models/', '')
            model_name = '/'.join(model_name.split('_', 1))

        fewshot = data["hyperparameters"]["fewshot"]
        key = f"{corpus}|{task}|{fewshot}"
        runs[model_name, corpus, task, fewshot].append((file_path, configuration(data)))

        if "ner" in task or "pos" in task:
            results[model_name][key]["overall_f1"].append(data["metrics"]["overall_f1"])
            results[model_name][key]["overall_accuracy"].append(data["metrics"]["overall_accuracy"])

        elif "cls" in task:
            results[model_name][key]["macro_f1"].append(data["metrics"]["macro avg"]["f1-score"])
            results[model_name][key]["weighted_f1"].append(data["metrics"]["weighted avg"]["f1-score"])

        elif "mcqa" in task:
            results[model_name][key]["hamming_score"].append(data["metrics"]["hamming_score"])
            results[model_name][key]["exact_match"].append(data["metrics"]["exact_match"])

        elif "regr" in task:
            results[model_name][key]["edrm"].append(data["metrics"]["EDRM"])
            results[model_name][key]["spearman_correlation_coef"].append(data["metrics"]["spearman_correlation_coef"])

    # The scores of runs trained in different ways must not be averaged
    problems = {}
    for group in sorted(runs, key=str):
        different = differences(runs[group], args.ignore)
        if different:
            problems[group] = different
    if problems:
        sys.exit(report(problems))

    print('Dumping to stats/results.json')
    with open(os.path.join(STATS_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print("Averaging runs...")
    # avg_results = {}
    avg_results = defaultdict(lambda: defaultdict(dict))
    for model in results:
        for task in results[model]:
            if not task.endswith("|1.0"):
                continue

            for metric, values in results[model][task].items():
                avg = sum(values) / len(values)
                avg_results[model][task][metric] = avg

    print('Dumping to stats/overall_averaged_metrics.json')
    with open(os.path.join(STATS_DIR, "overall_averaged_metrics.json"), 'w') as f:
        json.dump(avg_results, f, indent=4)
