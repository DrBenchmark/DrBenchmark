import json

f = open("./stats/results.json","r")
data = json.load(f)
f.close()

for model in data:

    for dataset_task in data[model]:

        for metric in data[model][dataset_task]:

            if len(data[model][dataset_task][metric]) != 4:

                print(model, " -> ", dataset_task, " -> ", metric, " -> ", len(data[model][dataset_task][metric]))
