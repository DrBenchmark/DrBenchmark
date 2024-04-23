# SubwordAnalysis.py

import json
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

f_in = open("./models.txt","r")
models = [m for m in f_in.read().split("\n") if len(m) > 0]
f_in.close()

tasks = [
    {"model": "Dr-BERT/DEFT2019", "subset": None, "dataset": None, "data_path": "./recipes/deft2019/data/"},
    {"model": "Dr-BERT/DEFT2021", "subset": "cls", "dataset": None, "data_path": "./recipes/deft2021/data/"},
    {"model": "Dr-BERT/DEFT2021", "subset": "ner", "dataset": None, "data_path": "./recipes/deft2021/data/"},

    {"model": "Dr-BERT/QUAERO", "subset": "emea", "dataset": None, "data_path": "./recipes/quaero/data/"},
    {"model": "Dr-BERT/QUAERO", "subset": "medline", "dataset": None, "data_path": "./recipes/quaero/data/"},
    {"model": "Dr-BERT/MANTRAGSC", "subset": "fr_emea", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "Dr-BERT/MANTRAGSC", "subset": "fr_medline", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "Dr-BERT/MANTRAGSC", "subset": "fr_patents", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "Dr-BERT/FrenchMedMCQA", "subset": None, "dataset": None, "data_path": "./recipes/frenchmedmcqa/data/"},
    {"model": "Dr-BERT/MORFITT", "subset": None, "dataset": None, "data_path": "./recipes/morfitt/data/"},
    {"model": "Dr-BERT/E3C", "subset": "French_clinical", "dataset": None, "data_path": "./recipes/e3c/data/"},
    {"model": "Dr-BERT/E3C", "subset": "French_temporal", "dataset": None, "data_path": "./recipes/e3c/data/"},
    {"model": "Dr-BERT/CLISTER", "subset": None, "dataset": None, "data_path": "./recipes/clister/data/"},
    {"model": "Dr-BERT/DEFT2020", "subset": "task_1", "dataset": None, "data_path": "./recipes/deft2020/data/"},
    {"model": "Dr-BERT/DEFT2020", "subset": "task_2", "dataset": None, "data_path": "./recipes/deft2020/data/"},
    {"model": "Dr-BERT/DiaMED", "subset": None, "dataset": None, "data_path": "./recipes/diamed/data/"},
    {"model": "Dr-BERT/PxCorpus", "subset": None, "dataset": None, "data_path": "./recipes/pxcorpus/data/"},

    {"model": "Dr-BERT/ESSAI", "subset": "pos", "dataset": None, "data_path": "./recipes/essai/data/"},
    {"model": "Dr-BERT/ESSAI", "subset": "ner_neg", "dataset": None, "data_path": "./recipes/essai/data/"},
    {"model": "Dr-BERT/ESSAI", "subset": "ner_spec", "dataset": None, "data_path": "./recipes/essai/data/"},
    {"model": "Dr-BERT/ESSAI", "subset": "cls", "dataset": None, "data_path": "./recipes/essai/data/"},

    {"model": "Dr-BERT/CAS", "subset": "pos", "dataset": None, "data_path": "./recipes/cas/data/"},
    {"model": "Dr-BERT/CAS", "subset": "ner_neg", "dataset": None, "data_path": "./recipes/cas/data/"},
    {"model": "Dr-BERT/CAS", "subset": "ner_spec", "dataset": None, "data_path": "./recipes/cas/data/"},
    {"model": "Dr-BERT/CAS", "subset": "cls", "dataset": None, "data_path": "./recipes/cas/data/"},
]

mapping = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
    "Dr-BERT/DrBERT-7GB": "DrBERT 7GB",
    "Dr-BERT/DrBERT-4GB-CP-PubMedBERT": "DrBERT CP PubMedBERT",
    "camembert-base": "CamemBERT",
    "almanach/camemberta-base": "CamemBERTa",
    "almanach/camembert-bio-base": "CamemBERT-BIO",
    "flaubert/flaubert_base_uncased": "FlauBERT",
    "emilyalsentzer/Bio_ClinicalBERT": "ClinicalBERT",
    "xlm-roberta-base": "XLM-RoBERTa",
    "distilbert-base-uncased": "DistilBERT",
}

matrix_avg_tokens_per_word = {f"{t['model']}-{t['subset']}": {m: [] for m in models} for t in tasks}

for m in models:

    print(f">> {m}")

    tokenizer = AutoTokenizer.from_pretrained(m)

    for task in tasks:

        if task['dataset'] == None:
            task['dataset'] = load_dataset(task['model'], task['subset'], data_dir=task['data_path'])["test"]

        t_key = f"{task['model']}-{task['subset']}"
        print(f">> {t_key}")

        for e in task['dataset']:

            # print(e)
            
            if task["model"].lower().find("quaero") != -1 or task["model"].lower().find("e3c") != -1 or task["model"].lower().find("mantragsc") != -1 or task["model"].lower().find("essai") != -1 or task["model"].lower().find("cas") != -1:
                
                if len(e["tokens"]) == 0:
                    nbr_tokens = 0
                else:
                    output = tokenizer(list(e["tokens"]), is_split_into_words=True)["input_ids"]
                    nbr_tokens = float(len(output) / len(e["tokens"]))
                    # print(float(len(output)))
                    # print(len(e["tokens"]))
            
            if task["model"].lower().find("frenchmedmcqa") != -1:
                text = f"{e['question']} {tokenizer.sep_token} " + f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in ["a","b","c","d","e"]])
                tokens = text.split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("morfitt") != -1:
                tokens = e['abstract'].split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))
            
            if task["model"].lower().find("deft") != -1 and task["subset"] == "task_1":
                text = f"{tokenizer.cls_token} {e['source']} {tokenizer.sep_token}  {e['cible']} {tokenizer.eos_token}"
                tokens = text.split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))
            
            if task["model"].lower().find("deft") != -1 and task["subset"] == "task_2":
                text = f"{e['source']} {tokenizer.sep_token} (1) {e['cible_1']} {tokenizer.sep_token} (2) {e['cible_2']} {tokenizer.sep_token} (3) {e['cible_3']}"
                tokens = text.split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("clister") != -1:
                text = f"{e['text_1']} {tokenizer.sep_token} {e['text_2']}"
                tokens = text.split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("diamed") != -1:
                text = f"{e['clinical_case']}"
                tokens = text.split(" ")
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("pxcorpus") != -1:
                tokens = e['tokens']
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("deft2019") != -1:
                tokens = e['tokens']
                if len(tokens) <= 0:
                    continue
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("deft2021") != -1 and task["subset"] == "ner":
                tokens = e['tokens']
                if len(tokens) <= 0:
                    continue
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            if task["model"].lower().find("deft2021") != -1 and task["subset"] == "cls":
                tokens = e['text'].split(" ")
                if len(tokens) <= 0:
                    continue
                output = tokenizer(list(tokens), is_split_into_words=True)["input_ids"]
                nbr_tokens = float(len(output) / len(tokens))

            matrix_avg_tokens_per_word[t_key][m].append(nbr_tokens)
            del nbr_tokens

with open("./stats/tokens.json", 'w') as f:
    json.dump(matrix_avg_tokens_per_word, f, indent=4)
print("JSON file saved!")

first_row = " & " + " & ".join([mapping[m] for m in models]) + "\\\\"
print(first_row)

for t in list(matrix_avg_tokens_per_word.keys()):

    print(f"{t.replace('Dr-BERT/','').replace('-',' ').replace('_',' ').replace('None','')} & ", end="")

    values = []

    for m in matrix_avg_tokens_per_word[t]:

        v = sum(matrix_avg_tokens_per_word[t][m]) / len(matrix_avg_tokens_per_word[t][m])
        values.append(str(round(v,2)))

    print(" & ".join(values) + " \\\\", end="")    
    print()
