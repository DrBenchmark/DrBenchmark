import json

import torch
import numpy as np

from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

f_in = open("./models.txt","r")
models = f_in.read().split("\n")
f_in.close()

tasks = [
    {"model": "DrBenchmark/MORFITT", "subset": None, "dataset": None, "data_path": "./recipes/morfitt/data/"},
    {"model": "DrBenchmark/QUAERO", "subset": "emea", "dataset": None, "data_path": "./recipes/quaero/data/"},
    {"model": "DrBenchmark/QUAERO", "subset": "medline", "dataset": None, "data_path": "./recipes/quaero/data/"},
    {"model": "DrBenchmark/MANTRAGSC", "subset": "fr_emea", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "DrBenchmark/MANTRAGSC", "subset": "fr_medline", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "DrBenchmark/MANTRAGSC", "subset": "fr_patents", "dataset": None, "data_path": "./recipes/mantragsc/data/"},
    {"model": "DrBenchmark/FrenchMedMCQA", "subset": None, "dataset": None, "data_path": "./recipes/frenchmedmcqa/data/"},
    {"model": "DrBenchmark/E3C", "subset": "French", "dataset": None, "data_path": "./recipes/e3c/data/"},
    {"model": "DrBenchmark/CLISTER", "subset": None, "dataset": None, "data_path": "./recipes/clister/data/"},
    {"model": "DrBenchmark/ESSAI", "subset": None, "dataset": None, "data_path": "./recipes/essai/data/"},
    {"model": "DrBenchmark/CAS", "subset": None, "dataset": None, "data_path": "./recipes/cas/data/"},
    {"model": "DrBenchmark/DEFT2020", "subset": "task_1", "dataset": None, "data_path": "./recipes/deft2020/data/"},
    {"model": "DrBenchmark/DEFT2020", "subset": "task_2", "dataset": None, "data_path": "./recipes/deft2020/data/"},
    {"model": "DrBenchmark/DiaMED", "subset": None, "dataset": None, "data_path": "./recipes/diamed/data/"},
    {"model": "DrBenchmark/PxCorpus", "subset": None, "dataset": None, "data_path": "./recipes/pxcorpus/data/"},
    {"model": "DrBenchmark/DEFT2019", "subset": None, "dataset": None, "data_path": "./recipes/deft2019/data/"},
    {"model": "DrBenchmark/DEFT2021", "subset": "cls", "dataset": None, "data_path": "./recipes/deft2021/data/"},
    {"model": "DrBenchmark/DEFT2021", "subset": "ner", "dataset": None, "data_path": "./recipes/deft2021/data/"},
]

sentences_perplexity = {f"{task['model']}-{task['subset']}": {m: [] for m in models} for task in tasks}

def ppl(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt', truncation=True, max_length=512)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    return np.exp(loss.item())

for model_name in models:

    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for task in tasks:

        if task['dataset'] == None:
            task['dataset'] = load_dataset(task['model'], task['subset'], data_dir=task['data_path'])["test"]

        t_key = f"{task['model']}-{task['subset']}"
        print(f">> {t_key}")

        for e in task['dataset']:

            if task["model"].lower().find("quaero") != -1 or task["model"].lower().find("e3c") != -1 or task["model"].lower().find("mantragsc") != -1 or task["model"].lower().find("essai") != -1 or task["model"].lower().find("cas") != -1:
                sentence = " ".join(e["tokens"])
            
            if task["model"].lower().find("frenchmedmcqa") != -1:
                sentence = f"{e['question']} {tokenizer.sep_token} " + f" {tokenizer.sep_token} ".join([e[f"answer_{letter}"] for letter in ["a","b","c","d","e"]])

            if task["model"].lower().find("morfitt") != -1:
                sentence = e['abstract']
            
            if task["model"].lower().find("deft") != -1 and task["subset"] == "task_1":
                sentence = f"{tokenizer.cls_token} {e['source']} {tokenizer.sep_token}  {e['cible']} {tokenizer.eos_token}"
            
            if task["model"].lower().find("deft") != -1 and task["subset"] == "task_2":
                sentence = f"{e['source']} {tokenizer.sep_token} (1) {e['cible_1']} {tokenizer.sep_token} (2) {e['cible_2']} {tokenizer.sep_token} (3) {e['cible_3']}"

            if task["model"].lower().find("clister") != -1:
                sentence = f"{e['text_1']} {tokenizer.sep_token} {e['text_2']}"

            perplexity = ppl(sentence=sentence, model=model, tokenizer=tokenizer)
            print(perplexity)
            sentences_perplexity[t_key][model_name].append(perplexity)
            del sentence
            del perplexity

with open("./stats/perplexity_raw.json", 'w') as f:
    json.dump(sentences_perplexity, f, indent=4)
print("JSON file saved!")

for t_key in sentences_perplexity:
    for model_name in sentences_perplexity[t_key]:
        sentences_perplexity[t_key][model_name] = sum(sentences_perplexity[t_key][model_name]) / len(sentences_perplexity[t_key][model_name])

with open("./stats/perplexity_avg.json", 'w') as f:
    json.dump(sentences_perplexity, f, indent=4)
print("JSON file saved!")
