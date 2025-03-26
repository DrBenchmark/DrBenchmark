# python BiasMaskFilling.py

import os
import json
from collections import Counter

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer

tasks = [
    {"corpus": "DrBenchmark/QUAERO", "subset": "emea", "dataset": None, "data_path": "./recipes/quaero/data/", "task": None},
    {"corpus": "DrBenchmark/QUAERO", "subset": "medline", "dataset": None, "data_path": "./recipes/quaero/data/", "task": None},
    {"corpus": "DrBenchmark/MANTRAGSC", "subset": "fr_emea", "dataset": None, "data_path": "./recipes/mantragsc/data/", "task": None},
    {"corpus": "DrBenchmark/MANTRAGSC", "subset": "fr_medline", "dataset": None, "data_path": "./recipes/mantragsc/data/", "task": None},
    {"corpus": "DrBenchmark/MANTRAGSC", "subset": "fr_patents", "dataset": None, "data_path": "./recipes/mantragsc/data/", "task": None},
    {"corpus": "DrBenchmark/FrenchMedMCQA", "subset": None, "dataset": None, "data_path": "./recipes/frenchmedmcqa/data/", "task": 1},
    {"corpus": "DrBenchmark/FrenchMedMCQA", "subset": None, "dataset": None, "data_path": "./recipes/frenchmedmcqa/data/", "task": 2},
    {"corpus": "DrBenchmark/MORFITT", "subset": None, "dataset": None, "data_path": "./recipes/morfitt/data/", "task": None},
    {"corpus": "DrBenchmark/E3C", "subset": "French_clinical", "dataset": None, "data_path": "./recipes/e3c/data/", "task": None},
    {"corpus": "DrBenchmark/E3C", "subset": "French_temporal", "dataset": None, "data_path": "./recipes/e3c/data/", "task": None},
    {"corpus": "DrBenchmark/CLISTER", "subset": None, "dataset": None, "data_path": "./recipes/clister/data/", "task": None},
    {"corpus": "DrBenchmark/DEFT2020", "subset": "task_1", "dataset": None, "data_path": "./recipes/deft2020/data/", "task": None},
    {"corpus": "DrBenchmark/DEFT2020", "subset": "task_2", "dataset": None, "data_path": "./recipes/deft2020/data/", "task": None},
    {"corpus": "DrBenchmark/PxCorpus", "subset": None, "dataset": None, "data_path": "./recipes/pxcorpus/data/", "task": 1},
    {"corpus": "DrBenchmark/PxCorpus", "subset": None, "dataset": None, "data_path": "./recipes/pxcorpus/data/", "task": 2},
    {"corpus": "DrBenchmark/DiaMED", "subset": None, "dataset": None, "data_path": "./recipes/diamed/data/", "task": None},
    {"corpus": "DrBenchmark/DEFT2019", "subset": None, "dataset": None, "data_path": "./recipes/deft2019/data/", "task": None},
    {"corpus": "DrBenchmark/DEFT2021", "subset": "cls", "dataset": None, "data_path": "./recipes/deft2021/data/", "task": None},
    {"corpus": "DrBenchmark/DEFT2021", "subset": "ner", "dataset": None, "data_path": "./recipes/deft2021/data/", "task": None},

    {"corpus": "DrBenchmark/CAS", "subset": "pos", "dataset": None, "data_path": "./recipes/cas/data/", "task": None},
    {"corpus": "DrBenchmark/CAS", "subset": "cls", "dataset": None, "data_path": "./recipes/cas/data/", "task": None},
    {"corpus": "DrBenchmark/CAS", "subset": "ner_neg", "dataset": None, "data_path": "./recipes/cas/data/", "task": None},
    {"corpus": "DrBenchmark/CAS", "subset": "ner_spec", "dataset": None, "data_path": "./recipes/cas/data/", "task": None},

    {"corpus": "DrBenchmark/ESSAI", "subset": "pos", "dataset": None, "data_path": "./recipes/essai/data/", "task": None},
    {"corpus": "DrBenchmark/ESSAI", "subset": "cls", "dataset": None, "data_path": "./recipes/essai/data/", "task": None},
    {"corpus": "DrBenchmark/ESSAI", "subset": "ner_neg", "dataset": None, "data_path": "./recipes/essai/data/", "task": None},
    {"corpus": "DrBenchmark/ESSAI", "subset": "ner_spec", "dataset": None, "data_path": "./recipes/essai/data/", "task": None},
]

tasks_classes = {f"{t['corpus']}-{t['subset']}-{t['task']}": {
    "train": [],
    "validation": [],
    "test": [],
} for t in tasks}

for subset in ["train","validation","test"]:

    for task in tasks:

        dataset = load_dataset(task['corpus'], task['subset'], data_dir=task['data_path'])[subset]

        t_key = f"{task['corpus']}-{task['subset']}-{task['task']}"
        print(f">> {t_key}")

        for e in dataset:

            # print(e)
            
            if task["corpus"].lower().find("quaero") != -1 or task["corpus"].lower().find("mantragsc") != -1:

                label_list = dataset.features[f"ner_tags"].feature.names
                current_classes = [label_list[n] for n in e['ner_tags']]

            if task["corpus"].lower().find("e3c") != -1:

                label_list = dataset.features[f"ner_tags"].feature.names
                current_classes = [label_list[n] for n in e['ner_tags']]
            
            if (task["corpus"].lower().find("essai") != -1 or task["corpus"].lower().find("cas") != -1) and task["subset"] == "pos":

                label_list = dataset.features["pos_tags"][0].names
                current_classes = [label_list[n] for n in e['pos_tags']]

            if (task["corpus"].lower().find("essai") != -1 or task["corpus"].lower().find("cas") != -1) and task["subset"].find("ner_") != -1:

                label_list = dataset.features["ner_tags"][0].names
                current_classes = [label_list[n] for n in e['ner_tags']]
            
            if (task["corpus"].lower().find("essai") != -1 or task["corpus"].lower().find("cas") != -1) and task["subset"] == "cls":

                label_list = dataset.features["label"].names
                current_classes = [label_list[e['label']]]
            
            if task["corpus"].lower().find("frenchmedmcqa") != -1 and task["task"] == 1:

                label_list = dataset.features["correct_answers"].feature.names
                current_classes = [label_list[n] for n in e['correct_answers']]
            
            if task["corpus"].lower().find("frenchmedmcqa") != -1 and task["task"] == 2:

                label_list = dataset.features["number_correct_answers"].names
                current_classes = [label_list[n] for n in [e['number_correct_answers']]]

            if task["corpus"].lower().find("morfitt") != -1:

                label_list = dataset.features["specialities"].feature.names
                current_classes = [label_list[n] for n in e['specialities']]

            if task["corpus"].lower().find("deft2021") != -1 and task["subset"] == "cls":

                label_list = dataset.features["specialities"].feature.names
                current_classes = [label_list[n] for n in e['specialities']]

            if task["corpus"].lower().find("deft2021") != -1 and task["subset"] == "ner":

                label_list = dataset.features[f"ner_tags"].feature.names
                current_classes = [label_list[n] for n in e['ner_tags']]

            if task["corpus"].lower().find("deft2019") != -1:

                label_list = dataset.features[f"ner_tags"].feature.names
                current_classes = [label_list[n] for n in e['ner_tags']]

            if task["corpus"].lower().find("pxcorpus") != -1 and task["task"] == 1:

                label_list = dataset.features[f"ner_tags"].feature.names
                current_classes = [label_list[n] for n in e['ner_tags']]

            if task["corpus"].lower().find("pxcorpus") != -1 and task["task"] == 2:

                label_list = dataset.features["label"].names
                current_classes = [label_list[n] for n in [e['label']]]

            if task["corpus"].lower().find("diamed") != -1:

                label_list = dataset.features["icd-10"].names
                current_classes = [label_list[n] for n in [e['icd-10']]]
            
            if task["corpus"].lower().find("deft2020") != -1 and task["subset"] == "task_1":
                current_classes = [e['moy']]
            
            if task["corpus"].lower().find("deft2020") != -1 and task["subset"] == "task_2":

                label_list = dataset.features["correct_cible"].names
                current_classes = [label_list[n] for n in [e['correct_cible']]]

            if task["corpus"].lower().find("clister") != -1:
                current_classes = [e['label']]

            tasks_classes[t_key][subset].extend(current_classes)
            del current_classes

os.makedirs("./stats/distributions", exist_ok=True)

for subset in ["train","validation","test"]:
    
    print(subset)

    for task in tasks:

        t_key = f"{task['corpus']}-{task['subset']}-{task['task']}"
        print(t_key)

        if (task["corpus"].lower().find("deft2020") != -1 and task["subset"] == "task_1") or (task["corpus"].lower().find("clister") != -1):
            sns.boxplot(x=tasks_classes[t_key][subset])            
            plt.title('Distribution of the values', fontsize=16)
            plt.xlabel('Frequencies')
            plt.ylabel('Classes')
            plt.savefig(f"./stats/distributions/{t_key.replace('/','_')}.png", bbox_inches='tight', dpi=200)
            plt.cla()
            plt.clf()

        else:
            tasks_classes[t_key][subset] = Counter(tasks_classes[t_key][subset]).most_common(9999)
            tasks_classes[t_key][subset] = sorted(tasks_classes[t_key][subset], key=lambda tup: tup[0])

            df = pd.DataFrame({
                'classes': [s[0] for s in tasks_classes[t_key][subset] if s[0] != "O"],
                'frequencies': [s[1] for s in tasks_classes[t_key][subset] if s[0] != "O"]
            })
            sns.barplot(x=df.frequencies, y=df.classes, orient='h')
            plt.title('Distribution of classes', fontsize=16)
            plt.xlabel('Frequencies')
            plt.ylabel('Classes')
            plt.savefig(f"./stats/distributions/{t_key.replace('/','_')}.png", bbox_inches='tight', dpi=200)
            plt.cla()
            plt.clf()

        print(tasks_classes[t_key][subset])

with open("./stats/classes_distribution.json", 'w') as f:
    json.dump(tasks_classes, f, indent=4)
print("JSON file saved!")
