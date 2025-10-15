import json


with open("./stats/results.json") as f:
    data = json.load(f)

for model in data:

    for dataset_task in data[model]:

        for metric in data[model][dataset_task]:

            if len(data[model][dataset_task][metric]) != 4:

                print(model, " -> ", dataset_task, " -> ", metric, " -> ", len(data[model][dataset_task][metric]))
