"""List the tasks that do not have the expected number of runs, for each model.

The scores of the runs are read in `stats/results.json`, which `ComputeMetrics.py` writes.
"""

import os
import json
import argparse

from paths import STATS_DIR
from protocol import NB_RUNS


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nb-run", type=int, required=False, default=NB_RUNS,
                        help=f"Number of runs expected for each task and each model (default: {NB_RUNS}).")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    with open(os.path.join(STATS_DIR, "results.json")) as f:
        data = json.load(f)

    for model in data:

        for dataset_task in data[model]:

            for metric in data[model][dataset_task]:

                if len(data[model][dataset_task][metric]) != args.nb_run:

                    print(model, " -> ", dataset_task, " -> ", metric, " -> ", len(data[model][dataset_task][metric]))
