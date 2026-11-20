import os
import sys
import json

import pandas as pd
from liquid import parse

from paths import REPO_DIR, STATS_DIR
from protocol import NB_RUNS

sys.path.append(REPO_DIR)
from run import task2script

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
# recipes/ and logs/ are looked for in the directory the job is submitted from
if [ ! -d recipes ]; then
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


if __name__ == '__main__':
    import argparse

    def arguments():
        parser = argparse.ArgumentParser(
            description='Write a script for SLURM that does the runs which are missing, for each task and each model '
                        'of `stats/results.json`.')
        parser.add_argument("--nb-run", type=int, required=False, default=NB_RUNS,
                            help=f"Number of runs expected for each task and each model (default: {NB_RUNS}).")
        parser.add_argument("--jeanzay", action='store_true',
                            help="Use jean-zay's slurm variables.")
        parser.add_argument("--filter", type=str, nargs='+', required=False, default=[],
                            help="Filter out jobs containing any of the texts (similar to echo \"$JOBS\" | grep -v A | grep -v B).")
        args = parser.parse_args()

        return args

    args = arguments()

    with open(os.path.join(STATS_DIR, 'results.json')) as f:
        res = json.load(f)

    df = pd.DataFrame(
        [[model, *task.split('|'), metric, score * 100]
            for model, tasks in res.items()
            for task, metrics in tasks.items()
            for metric, scores in metrics.items()
            for score in scores],
        columns='model dataset task fewshot metric score'.split()
    )

    # Number of runs of each task for each model: the one of its metric that has the most scores
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].apply(len)
    nb_runs = nb_runs.unstack(4).fillna(0).apply(max, axis=1)
    # Add all datasets and tasks
    all_models = df['model'].unique()
    dat_tasks = [e.split('-') for e in task2script]
    nb_runs = nb_runs.reindex([(m, d, t, '1.0') for m in all_models for d, t in dat_tasks]).fillna(0)
    # Number of runs that are missing
    nb_runs = (args.nb_run - nb_runs[nb_runs < args.nb_run]).astype(int)
    nb_runs = nb_runs.reset_index().drop_duplicates()
    JOBS = []
    for i, r in nb_runs.iterrows():
        corpus = r['dataset']
        task = r['task']
        model = r['model']
        comm = f'cd recipes/{corpus}/scripts/ ; ' + task2script[f'{corpus}-{task}'].format(model_name=model)
        if any(f in comm for f in args.filter):
            continue
        JOBS += [comm] * r[0]

    if not JOBS:
        # An empty job array is not valid for SLURM (--array=0--1)
        sys.exit('No missing run: there is no SLURM script to generate.')

    # `parse(...).render(mapping)` works with python-liquid 2 and 3, unlike `render(..., **variables)`
    print(parse(slurm_template).render({'JOBS': JOBS, 'JEANZAY': args.jeanzay}))
