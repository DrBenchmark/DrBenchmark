import os
import json

from paths import STATS_DIR


with open(os.path.join(STATS_DIR, "results.json")) as f:
    data = json.load(f)

for model in data:

    for dataset_task in data[model]:

        for metric in data[model][dataset_task]:

            if len(data[model][dataset_task][metric]) != 4:

                print(model, " -> ", dataset_task, " -> ", metric, " -> ", len(data[model][dataset_task][metric]))
