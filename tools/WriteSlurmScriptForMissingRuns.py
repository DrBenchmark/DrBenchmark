"""Write a script for SLURM that does the runs which are missing.

The benchmark expects 4 runs (`--nb-run`) on the whole training set, for each task of `run.py`
and each model of `models.txt` (`--models`). The runs are read in `recipes/<recipe>/runs/`,
and the script that is printed has one job for each run that is missing.

A job is a run of `run.py` with a seed: one of the first seeds, from 0, that no run of the
task for the model has. The runs that `run.py` computed are then not asked for again, and the
ones that a launcher computed with a seed it drew are counted too.

The script has to be submitted from the root of the repository.
"""

import sys
import shlex
import argparse

from liquid import parse

from models_and_datasets import models_to_evaluate
from number_of_runs import NB_RUNS
from repository_paths import REPO_DIR

# `run.py` has the tasks, and `predict.py` reads the runs: both are at the root of the repository
sys.path.insert(0, REPO_DIR)
import predict  # noqa: E402
from run import task2script  # noqa: E402

slurm_template = """#!/bin/bash
#SBATCH --job-name=DrBenchmark
#SBATCH --ntasks=1             # Total MPI process per job
#SBATCH --ntasks-per-node=1    # Total MPI process per node per job
#SBATCH --hint=nomultithread   # 1 MPI process per phisical core (no hyperthreading)
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/%x_%A_%a.err
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=0-{{ JOBS | size | minus: 1 }}%10      # one job per missing run, 10 jobs max running at the same time
{% if JEANZAY -%}
#SBATCH --account=<ACCOUNT>@v100
{% else -%}
#SBATCH --partition=gpu
#SBATCH --constraint='GPURAM_Min_12GB'
{% endif %}
# run.py, recipes/ and logs/ are looked for in the directory the job is submitted from
if [ ! -f run.py ] || [ ! -d recipes ]; then
    echo "This script has to be submitted from the root of the DrBenchmark repository, not from $PWD" >&2
    exit 1
fi

{% if JEANZAY -%}
module purge
source ~/.bashrc

{% endif -%}

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

JOBS+=({% for j in JOBS %}
    "{{j}}"
{%- endfor %}
)

# Select which job to run
CURRENT=$SLURM_ARRAY_TASK_ID
COMMAND=${JOBS[$CURRENT]}
echo "index: $CURRENT, value: $COMMAND"
# Run the job
eval "$COMMAND"
"""


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nb-run", type=int, required=False, default=NB_RUNS,
                        help=f"Number of runs expected for each task and each model (default: {NB_RUNS}).")
    parser.add_argument("--models", type=str, nargs='+', required=False, default=None,
                        help="Name or path of the models that have to have their runs (default: the models of models.txt).")
    parser.add_argument("--jeanzay", action='store_true',
                        help="Use jean-zay's slurm variables.")
    parser.add_argument("--filter", type=str, nargs='+', required=False, default=[],
                        help="Filter out jobs containing any of the texts (similar to echo \"$JOBS\" | grep -v A | grep -v B).")
    return parser.parse_args()


def missing_runs(runs, tasks, models, nb_run):
    """The runs that are missing, as `run.py` does them: (task, model, seed), task by task then model by model.

    The runs of a task for a model are counted whatever their seeds. The ones that are
    missing get the first seeds, from 0, that none of them has.
    """
    missing = []
    for task in tasks:
        for model in models:
            seeds = [run.seed for run in runs if run.task == task and run.fewshot == 1.0 and predict.is_model(run, model)]
            free = (seed for seed in range(nb_run + len(seeds)) if seed not in seeds)
            missing += [(task, model, next(free)) for _ in range(nb_run - len(seeds))]
    return missing


def main():
    args = arguments()
    if args.models is None:
        try:
            args.models = models_to_evaluate()
        except FileNotFoundError:
            sys.exit(f"`models.txt` is not in {REPO_DIR}: give the models with --models.")

    # The tasks in the order `run.py --tasks all` does them
    jobs = [
        f"python run.py --tasks {task} --models {shlex.quote(model)} --nb-run 1 --seed {seed}"
        for task, model, seed in missing_runs(predict.read_runs(), sorted(task2script), args.models, args.nb_run)
    ]
    jobs = [job for job in jobs if not any(text in job for text in args.filter)]
    if not jobs:
        # An empty job array is not valid for SLURM (--array=0--1)
        sys.exit('No missing run: there is no SLURM script to generate.')

    # `parse(...).render(mapping)` works with python-liquid 2 and 3, unlike `render(..., **variables)`
    print(parse(slurm_template).render({'JOBS': jobs, 'JEANZAY': args.jeanzay}))


if __name__ == '__main__':
    main()
