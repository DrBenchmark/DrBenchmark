import os
import json

import matplotlib.pyplot as plt


with open("./stats/results.json") as f:
    results = json.load(f)

avg_results = {}

for model in results:

    # print(model)

    if model not in avg_results:
        avg_results[model] = {}

    for task in results[model]:

        if task not in avg_results[model]:
            avg_results[model][task] = {}

        # print(task)

        for metric in results[model][task].keys():

            avg = sum(results[model][task][metric]) / len(results[model][task][metric])
            # print(">> ", metric , "-", avg)

            if metric not in avg_results[model][task]:
                avg_results[model][task][metric] = -1

            avg_results[model][task][metric] = avg

bottom = [0.25, 0.50, 0.75, 1.0]
# bottom = [0.0, 0.25, 0.50, 0.75, 1.0]

models_names = list(avg_results.keys())
results_fewshot = {}

for model in avg_results:

    for taskf in avg_results[model]:

        corpus, task, fewshot = taskf.split("|")
        idx_fewshot = bottom.index(float(fewshot))

        key = f"{corpus}|{task}"

        if key not in results_fewshot:
            results_fewshot[key] = {}

        if model not in results_fewshot[key]:
            results_fewshot[key][model] = [0.0] * len(bottom)
            # results_fewshot[key][model] = [0.0] * len(bottom)

        if task.find("regr") != -1 or task.find("regr") != -1:
            metric = float(f"{round(avg_results[model][taskf]['edrm'] * 100, 2)}")
            # metric = f"{round(avg_results[model][taskf]['edrm'], 2)}" + " / " + f"{round(avg_results[model][taskf]['spearman_correlation_coef'], 2)}"

        elif task.find("mcqa") != -1:
            metric = float(f"{round(avg_results[model][taskf]['hamming_score'] * 100, 2)}")
            # metric = f"{round(avg_results[model][taskf]['hamming_score'], 2)} / {round(avg_results[model][taskf]['exact_match'], 2)}"

        elif task.find("pos") != -1 or task.find("ner") != -1:
            metric = float(f"{round(avg_results[model][taskf]['overall_f1'] * 100, 2)}")
            # metric = f"{round(avg_results[model][taskf]['overall_f1'], 2)}"

        else:
            metric = float(f"{round(avg_results[model][taskf]['weighted_f1'] * 100, 2)}")
            # metric = f"{round(avg_results[model][taskf]['weighted_f1'] * 100, 2)}"

        results_fewshot[key][model][idx_fewshot] = metric

mapping = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
    "microsoft/Biomednlp-PubmedBERT-base-uncased-abstract": "OLD-PubMedBERT",
    "Dr-BERT/DrBERT-7GB": "DrBERT-FS",
    "Dr-BERT/DrBERT-4GB-CP-PubMedBERT": "DrBERT-CP",
    "camembert-base": "CamemBERT",
    "almanach/camembert-base": "CamemBERT",
    "almanach/camemberta-base": "CamemBERTa",
    "almanach/camembert-bio-base": "CamemBERT-BIO",
    "flaubert/flaubert_base_uncased": "FlauBERT",
    "emilyalsentzer/Bio_ClinicalBERT": "ClinicalBERT",
    "xlm-roberta-base": "XLM-RoBERTa",
    "FacebookAI/xlm-roberta-base": "XLM-RoBERTa",
    "distilbert-base-uncased": "DistilBERT",
    "distilbert/distilbert-base-uncased": "DistilBERT",
}
mapping = {k.lower().replace('/', '_'): v for k, v in mapping.items()}

mapping_line = {
    "CamemBERTa": "-",
    "CamemBERT": "--",
    "CamemBERT-BIO": "-.",
    "DrBERT-FS": ":",
    "DrBERT-CP": "-.",
    "PubMedBERT": "-",
    "FlauBERT": "--",
    "XLM-RoBERTa": "-",
}

os.makedirs("./stats/fewshot", exist_ok=True)

for key in results_fewshot:

    corpus, task = key.split("|")

    min_val = 100
    max_val = 0

    for model in results_fewshot[key]:

        values = [float(v) for v in results_fewshot[key][model]]
        model_name = mapping[model.replace("../../../models/", "")]
        model_linestyle = mapping_line[model_name]

        if min(values) < min_val:
            min_val = min(values)

        if max(values) > max_val:
            max_val = max(values)

        plt.plot(bottom, values, label=model_name, linestyle=model_linestyle)

    plt.legend(loc="lower right", prop={'size': 15})

    if max_val + 5 > 100:
        max_val = 100
    else:
        max_val = max_val + 5

    plt.ylim(min_val - 5, max_val)
    plt.xticks(bottom, [0.25, 0.50, 0.75, 1.0])

    plt.tight_layout()

    output_name = f"./stats/fewshot/{corpus}@{task}.pdf"
    print(output_name)
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)
    plt.clf()
