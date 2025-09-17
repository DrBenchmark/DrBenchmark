import os
import json
import statistics
from glob import glob

path = "./recipes/"
dirs = [ f.path for f in os.scandir(path) if f.is_dir() ]
print(dirs)

def contains(p, t):
    return p.find(t) != -1

results = {}

for d in dirs:

    if "__pycache__" in d:
        continue

    # Skip mantragsc directories entirely
    if contains(d, "/mantragsc/") or d.endswith("mantragsc"):
        print(f"Skipping mantragsc directory: {d}")
        continue

    print(d)

    d_path = f"{d}/runs/"
    files = [ f.path for f in os.scandir(d_path) if f.is_file() and str(f).find(".json") != -1 ]
    print(files)

    for file_path in files:

        f_json = open(file_path, "r")
        data = json.load(f_json)
        f_json.close()

        print(file_path)
        corpus = ""
        task = ""

        # TASK
        if contains(file_path, "-cls-"):
            task = "cls"
        elif contains(file_path, "-French_clinical-"):
            task = "ner-clinical"
        elif contains(file_path, "-French_temporal-"):
            task = "ner-temporal"
        elif contains(file_path, "-fr_emea-"):
            task = "ner-fr_emea"
        elif contains(file_path, "-fr_medline-"):
            task = "ner-fr_medline"
        elif contains(file_path, "-fr_patents-"):
            task = "ner-fr_patents"
        elif contains(file_path, "-emea-"):
            task = "ner-emea"
        elif contains(file_path, "-medline-"):
            task = "ner-medline"
        elif contains(file_path, "-pos-"):
            task = "pos"
        elif contains(file_path, "-regression-"):
            task = "regr"
        elif contains(file_path, "-mcqa-"):
            task = "mcqa"
        elif contains(file_path, "-ner-"):
            task = "ner"
        elif contains(file_path, "-ner_neg-"):
            task = "ner-neg"
        elif contains(file_path, "-ner_spec-"):
            task = "ner-spec"

        # CORPUS
        if contains(file_path, "/cas/"):
            corpus = "cas"
        elif contains(file_path, "/clister/"):
            corpus = "clister"
        elif contains(file_path, "/deft2020/"):
            corpus = "deft2020"
        elif contains(file_path, "/e3c/"):
            corpus = "e3c"
        elif contains(file_path, "/essai/"):
            corpus = "essai"
        elif contains(file_path, "/frenchmedmcqa/"):
            corpus = "frenchmedmcqa"
        # elif contains(file_path, "/mantragsc/"):
        #     corpus = "mantragsc"  # REMOVED MANTRAGSC
        elif contains(file_path, "/morfitt/"):            
            corpus = "morfitt"
        elif contains(file_path, "/quaero/"):
            corpus = "quaero"
        elif contains(file_path, "/pxcorpus/"):
            corpus = "pxcorpus"
        elif contains(file_path, "/diamed/"):
            corpus = "diamed"
        elif contains(file_path, "/deft2019/"):
            corpus = "deft2019"
        elif contains(file_path, "/deft2021/"):
            corpus = "deft2021"

        # Skip mantragsc corpus just in case
        if corpus == "mantragsc":
            continue

        model_name = data["hyperparameters"]["model_name"]

        if model_name.find("../../../models/") == -1:
            continue

        fewshot = data["hyperparameters"]["fewshot"]
        key = f"{corpus}|{task}|{fewshot}"
        
        print(key + "-" + model_name)

        if model_name not in results:
            results[model_name] = {}

        if task.find("ner") != -1 or task.find("pos") != -1:

            if key not in results[model_name]:
                results[model_name][key] = {"overall_f1": [], "overall_accuracy": []}
            
            results[model_name][key]["overall_f1"].append(data["metrics"]["overall_f1"])
            results[model_name][key]["overall_accuracy"].append(data["metrics"]["overall_accuracy"])
        
        elif task.find("cls") != -1:

            if key not in results[model_name]:
                results[model_name][key] = {"macro_f1": [], "weighted_f1": []}
            
            results[model_name][key]["macro_f1"].append(data["metrics"]["macro avg"]["f1-score"])
            results[model_name][key]["weighted_f1"].append(data["metrics"]["weighted avg"]["f1-score"])

        elif task.find("mcqa") != -1:

            if key not in results[model_name]:
                results[model_name][key] = {"hamming_score": [], "exact_match": []}
            
            results[model_name][key]["hamming_score"].append(data["metrics"]["hamming_score"])
            results[model_name][key]["exact_match"].append(data["metrics"]["exact_match"])
        
        elif task.find("regr") != -1:

            if key not in results[model_name]:
                results[model_name][key] = {"edrm": [], "spearman_correlation_coef": []}
            
            results[model_name][key]["edrm"].append(data["metrics"]["EDRM"])
            results[model_name][key]["spearman_correlation_coef"].append(data["metrics"]["spearman_correlation_coef"])

with open("./stats/results.json", 'w') as f:
    json.dump(results, f, indent=4)

avg_results = {}

for model in results:
    
    print(model)

    if model not in avg_results:
        avg_results[model] = {}
    
    for task in results[model]:

        if not task.endswith("|1.0"):
            continue

        if task not in avg_results[model]:
            avg_results[model][task] = {}
        
        print(task)

        for metric in results[model][task].keys():

            values = results[model][task][metric]
            avg = sum(values) / len(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            num_runs = len(values)
            
            print(">> ", metric , "- avg:", avg, "std:", std, "runs:", num_runs)

            if metric not in avg_results[model][task]:
                avg_results[model][task][metric] = {}
            
            avg_results[model][task][metric] = {
                "avg": avg,
                "std": std,
                "num_runs": num_runs
            }

with open("./stats/overall_averaged_metrics.json", 'w') as f:
    json.dump(avg_results, f, indent=4)

# Print summary of runs BY MODEL
print("\n" + "="*60)
print("SUMMARY: Number of runs BY MODEL")
print("="*60)

model_run_summary = {}

for model in avg_results:
    model_run_summary[model] = {
        "total_runs": 0,
        "total_tasks": 0,
        "tasks_detail": {}
    }
    
    for task in avg_results[model]:
        # Get the number of runs from the first metric (they should all be the same for a given task)
        first_metric = list(avg_results[model][task].keys())[0]
        runs_for_task = avg_results[model][task][first_metric]["num_runs"]
        
        model_run_summary[model]["total_runs"] += runs_for_task
        model_run_summary[model]["total_tasks"] += 1
        model_run_summary[model]["tasks_detail"][task] = runs_for_task

# Sort models by total number of runs (descending)
sorted_models = sorted(model_run_summary.items(), key=lambda x: x[1]["total_runs"], reverse=True)

for model, summary in sorted_models:
    print(f"\n📊 Model: {model}")
    print(f"   Total runs: {summary['total_runs']}")
    print(f"   Total tasks: {summary['total_tasks']}")
    print(f"   Average runs per task: {summary['total_runs'] / summary['total_tasks']:.1f}")
    
    print("   Task breakdown:")
    for task, runs in summary["tasks_detail"].items():
        print(f"     • {task}: {runs} runs")

print(f"\n📈 OVERALL STATISTICS:")
print(f"   Total models: {len(model_run_summary)}")
total_all_runs = sum(summary["total_runs"] for summary in model_run_summary.values())
print(f"   Total runs across all models: {total_all_runs}")
avg_runs_per_model = total_all_runs / len(model_run_summary)
print(f"   Average runs per model: {avg_runs_per_model:.1f}")

with open("./stats/model_run_summary.json", 'w') as f:
    json.dump(model_run_summary, f, indent=4)

print(f"\n💾 Model run summary saved to: ./stats/model_run_summary.json")