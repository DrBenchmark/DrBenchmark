import os
import sys
import json

import pandas as pd
from liquid import render

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
#SBATCH --array=0-{{ JOBS | size }}%10      # 27tasks*8models*4runs = 864 jobs overall but 100 jobs max in the queue
{% if JEANZAY -%}
#SBATCH --account=<ACCOUNT>@v100
{% else -%}
#SBATCH --partition=gpu
#SBATCH --constraint='GPURAM_Min_12GB'
{% endif %}
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

custom_model_map = {
    "../../../models/almanach_camembert-bio-base": "almanach/camembert-bio-base",
    "../../../models/dr-bert_drbert-7gb": "Dr-BERT/DrBERT-7GB",
    "../../../models/dr-bert_drbert-7gb-large": "Dr-BERT/DrBERT-7GB-Large",
    "../../../models/flaubert_flaubert_base_cased": "flaubert/flaubert_base_cased",
    "../../../models/flaubert_flaubert_large_cased": "flaubert/flaubert_large_cased"
}


if __name__ == '__main__':
    import argparse

    def arguments():
        parser = argparse.ArgumentParser(
            description='Run selected DrBenchmark\'s task on selected models.')
        parser.add_argument("--jeanzay", action='store_true',
                            help="Use jean-zay's slurm variables.")
        parser.add_argument("--filter", type=str, nargs='+', required=False, default=[],
                            help="Filter out jobs containing any of the texts (similar to echo \"$JOBS\" | grep -v A | grep -v B).")
        args = parser.parse_args()

        return args

    args = arguments()

    with open('stats/results.json') as f:
        res = json.load(f)

    df = pd.DataFrame(
        [[model, *task.split('|'), metric, score * 100]
            for model, tasks in res.items()
            for task, metrics in tasks.items()
            for metric, scores in metrics.items()
            for score in scores],
        columns='model dataset task fewshot metric score'.split()
    )

    df['model'] = df['model'].map(lambda x: custom_model_map.get(x, x))

    # Only keep 4 runs and compute score's mean
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].apply(len)
    nb_runs = nb_runs.unstack(4).fillna(0).apply(max, axis=1)
    # Add all datasets and tasks
    all_models = df['model'].unique()
    dat_tasks = [e.split('-') for e in task2script]
    nb_runs = nb_runs.reindex([(m, d, t, '1.0') for m in all_models for d, t in dat_tasks]).fillna(0)
    nb_runs = (4 - nb_runs[nb_runs < 4]).astype(int)
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

    print(render(slurm_template, **{'JOBS': JOBS, 'JEANZAY': args.jeanzay}))
