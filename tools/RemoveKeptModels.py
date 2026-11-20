"""Remove models that the runs kept, to free disk space.

Each run keeps its best model in `recipes/<recipe>/save_models/`: about 440 MB for a model of
the size of BERT base. To predict for a task with a fine-tuned model, `predict.py` takes one
of them: the model of the run with the best score on the validation set, among the runs on
the whole training set (`python predict.py --list` shows it).

This tool removes the other ones: the models that `predict.py` does not take. With `--all`,
it removes every kept model. `--task` and `--model` say which runs are concerned.

Nothing is removed without `--remove`: the models that would be removed are listed, with
their size. The results of the runs are not touched: their scores and their predictions stay.
A run does not keep its model at all when it is given `--keep_model false`.
"""

import os
import sys
import shutil
import argparse

from repository_paths import REPO_DIR

# `predict.py`, at the root of the repository, reads the runs and chooses the one whose model is taken
sys.path.insert(0, REPO_DIR)
import predict  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, required=False, default=None,
                        help="Only the models of this task: one of the names of `python run.py --help`.")
    parser.add_argument("--model", type=str, required=False, default=None,
                        help="Only the models that were fine-tuned from this one: its name on the HuggingFace Hub, or its path.")
    parser.add_argument("--all", action="store_true",
                        help="Remove every kept model, the ones that `predict.py` takes too.")
    parser.add_argument("--remove", action="store_true",
                        help="Remove the models: without it, they are only listed.")
    return parser.parse_args()


def size_of(directory):
    """Number of bytes of the files of a directory."""
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(directory) for name in names
        if not os.path.islink(os.path.join(root, name))
    )


def as_size(size):
    for unit in ("bytes", "kB", "MB"):
        if size < 1000:
            return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"


def models_to_remove(runs, everything):
    """The runs whose kept model is removed, and the ones whose model stays."""
    # A run saves its model in a directory with this name: nothing else is removed, whatever its results say
    with_model = [run for run in runs if run.model_dir is not None and run.model_dir.endswith("_best_model")]
    if everything:
        return with_model, []
    taken = set()
    for name in {(run.task, run.model) for run in with_model}:
        best = predict.best_run([run for run in with_model if (run.task, run.model) == name])
        if best is not None:
            taken.add(best.path)
    return [run for run in with_model if run.path not in taken], [run for run in with_model if run.path in taken]


def main():
    args = arguments()
    removed, kept = models_to_remove(predict.read_runs(args.task, args.model), args.all)

    total = 0
    for run in removed:
        size = size_of(run.model_dir)
        total += size
        part = "" if run.fewshot == 1.0 else f", {run.fewshot:g} of the training set"
        print(f"{run.model_dir}  {as_size(size)}  ({run.task}, {run.model}, seed {run.seed}{part})")
        if args.remove:
            shutil.rmtree(run.model_dir)

    models = f"{len(removed)} model{'s' if len(removed) != 1 else ''}"
    stay = f"{len(kept)} model{'s' if len(kept) != 1 else ''} stay{'s' if len(kept) == 1 else ''}"
    if not removed:
        print(f"No model to remove: {stay}.")
    elif args.remove:
        print(f"{models} removed, {as_size(total)}: {stay}.")
    else:
        print(f"{models} to remove, {as_size(total)}: {stay}. Nothing was removed: give --remove to remove them.")


if __name__ == '__main__':
    main()
