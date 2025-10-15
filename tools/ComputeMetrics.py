import os
import json
from glob import glob
from collections import defaultdict

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x, *args, **kwargs):
        return x

# TASK
fp2task = {
    "-cls-": "cls",
    "-French_clinical-": "ner_clinical",
    "-French_temporal-": "ner_temporal",
    "-fr_emea-": "ner_emea",
    "-fr_medline-": "ner_medline",
    "-fr_patents-": "ner_patents",
    "-emea-": "ner_emea",
    "-medline-": "ner_medline",
    "-pos-": "pos",
    "-regression-": "regr",
    "-mcqa-": "mcqa",
    "-ner-": "ner",
    "-ner_neg-": "ner_neg",
    "-ner_spec-": "ner_spec"
}

# CORPUS
fp2corpus = {
    "/cas/": "cas",
    "/clister/": "clister",
    "/deft2020/": "deft2020",
    "/e3c/": "e3c",
    "/essai/": "essai",
    "/frenchmedmcqa/": "frenchmedmcqa",
    "/mantragsc/": "mantragsc",
    "/morfitt/": "morfitt",
    "/quaero/": "quaero",
    "/pxcorpus/": "pxcorpus",
    "/diamed/": "diamed",
    "/deft2019/": "deft2019",
    "/deft2021/": "deft2021",
}

if __name__ == '__main__':
    path = "recipes"

    # results: model -> corpus-task -> metric -> list
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    file_paths = glob(os.path.join(path, '*', 'runs', '*.json'))
    print("Reading run results...")
    for file_path in tqdm(file_paths):

        with open(file_path) as f_json:
            data = json.load(f_json)

        task = [v for k, v in fp2task.items() if k in file_path]
        assert len(task) != 0
        task = task[0]

        corpus = [v for k, v in fp2corpus.items() if k in file_path]
        assert len(corpus) != 0
        corpus = corpus[0]

        model_name = data["hyperparameters"]["model_name"]

        fewshot = data["hyperparameters"]["fewshot"]
        key = f"{corpus}|{task}|{fewshot}"

        if "ner" in task or "pos" in task:
            results[model_name][key]["overall_f1"].append(data["metrics"]["overall_f1"])
            results[model_name][key]["overall_accuracy"].append(data["metrics"]["overall_accuracy"])

        elif "cls" in task:
            results[model_name][key]["macro_f1"].append(data["metrics"]["macro avg"]["f1-score"])
            results[model_name][key]["weighted_f1"].append(data["metrics"]["weighted avg"]["f1-score"])

        elif "mcqa" in task:
            results[model_name][key]["hamming_score"].append(data["metrics"]["hamming_score"])
            results[model_name][key]["exact_match"].append(data["metrics"]["exact_match"])

        elif "regr" in task:
            results[model_name][key]["edrm"].append(data["metrics"]["EDRM"])
            results[model_name][key]["spearman_correlation_coef"].append(data["metrics"]["spearman_correlation_coef"])

    print('Dumping to stats/results.json')
    with open("./stats/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Averaging runs...")
    # avg_results = {}
    avg_results = defaultdict(lambda: defaultdict(dict))
    for model in results:
        for task in results[model]:
            if not task.endswith("|1.0"):
                continue

            for metric, values in results[model][task].items():
                avg = sum(values) / len(values)
                avg_results[model][task][metric] = avg

    print('Dumping to stats/overall_averaged_metrics.json')
    with open("./stats/overall_averaged_metrics.json", 'w') as f:
        json.dump(avg_results, f, indent=4)
