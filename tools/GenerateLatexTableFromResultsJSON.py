import json

f = open("./stats/overall_averaged_metrics.json")
data = json.load(f)
f.close()

f_in = open("./models.txt","r")
models = ["../../../models/" + m.lower().replace("/","_") for m in f_in.read().split("\n") if m.strip()]
f_in.close()

# Model mapping (unchanged)
mapping = {
    "almanach_camemberta-base": "CamemBERTa",
    "camembert-base": "CamemBERT",
    "almanach_camembert-bio-base": "CamemBERT-BIO",
    "dr-bert_drbert-7gb": "DrBERT 7GB",
    "dr-bert_drbert-4gb-cp-pubmedbert": "DrBERT CP PubMedBERT",
    "microsoft_biomednlp-pubmedbert-base-uncased-abstract-fulltext": "PubMedBERT",
    "microsoft_biomednlp-pubmedbert-base-uncased-abstract": "OLD-PubMedBERT",
    "flaubert_flaubert_base_uncased": "FlauBERT",
    "almanach_moderncamembert-cv2-base": "ModernCamemBERT",
    "doctobert_dynamic_mlm": "DoctoBERT",
    "doctobert_phase_1": "DoctoBERT p1",
    "doctobert_phase_2": "DoctoBERT p2",
    "thomas-sounack_bioclinical-modernbert-base": "Bio clinical ModernBERT",
    "doctobert_phase_2_gelu": "DoctoBERT p2 gelu",
    "doctobert_dynamic_mlm_gelu": "DoctoBERT p3 gelu",
    "ct-moderncamembert-decay-dynamic": "ModernCamemBERT NACHOS",
    "ct-bio-clinical-modernbert-decay-dynamic": "Bio Clinical ModernBERT NACHOS",
    "cp-moderncamembert-base-p1-54b-dynamic": "ModernCamembert Short context",
    "doctobert-exp-nachos-base-pretrain_750_128b_512_toks-lr-decay_8B_dynamic_mlm_prob-hf": "DoctoBERT Short Context",
    "doctobert-exp-fineweb2-base-pretrain-30b": "DoctoBERT 30b",
    "doctobert-exp-fineweb2-base-pretrain-30b-clinical": "DoctoBERT 30b clinical",
    "doctobert-exp-fineweb2-base-pretrain-30b-edu2": "DoctoBERT 30b edu 2",
    "doctobert-exp-fineweb2-base-pretrain-30b-edu3": "DoctoBERT 30b edu 3",
    "doctobert-exp-fineweb2-base-pretrain-30b-clinical-cs": "DoctoBERT 30b clinical cs",
    "doctobert-exp-fineweb2-base-pretrain-30b-clinical-edu2-ep9": "DoctoBERT 30b clinical edu 2",
    "doctobert-exp-fineweb2-base-pretrain-15b-clinical-vocab50k-ep3": "DoctoBERT 15b clinical vocab50k"
}

# Get all unique tasks present for all models
tasks = []
for model in models:
    if model in data and data[model]:
        tasks = list(data[model].keys())
        break

if not tasks:
    print("Error: No tasks found in any model data")
    exit()

# Organize tasks by corpus
from collections import defaultdict
corpus_tasks = defaultdict(list)
for t in tasks:
    corpus, task, fewshot = t.split("|")
    corpus_tasks[corpus].append((task, t))

output = []

output.append("""
\\begin{table*}[t!]
\\tiny
\\centering
\\begin{tabular}{|l|l|%s|}
\\hline
""" % ("c|"*len(models)))

output.append("\\textbf{Dataset} & \\textbf{Task} & " + " & ".join(["\\textbf{" + mapping.get(m.replace("../../../models/",""),m) + "}" for m in models]) + " \\\\ ")

for corpus in corpus_tasks:
    task_list = corpus_tasks[corpus]
    num_rows = len(task_list)
    corpus_written = False
    for idx, (task, tkey) in enumerate(task_list):
        # Skip tasks that don't exist for all models
        task_exists_for_all = all(m in data and tkey in data[m] for m in models)
        if not task_exists_for_all:
            continue

        runs_metrics = []

        if tkey.find("deft2020|regr") != -1 or tkey.find("clister|regr") != -1:
            # Regression (EDRM / Spearman)
            edrm_values, spear_values = [], []
            for m in models:
                edrm_avg = data[m][tkey]['edrm']['avg']
                edrm_std = data[m][tkey]['edrm']['std']
                spear_avg = data[m][tkey]['spearman_correlation_coef']['avg']
                spear_std = data[m][tkey]['spearman_correlation_coef']['std']
                edrm_values.append(edrm_avg)
                spear_values.append(spear_avg)
            edrm_best_idx = sorted(range(len(edrm_values)), key=lambda i: edrm_values[i], reverse=True)[0]
            edrm_second_idx = sorted(range(len(edrm_values)), key=lambda i: edrm_values[i], reverse=True)[1] if len(edrm_values) > 1 else None
            spear_best_idx = sorted(range(len(spear_values)), key=lambda i: spear_values[i], reverse=True)[0]
            spear_second_idx = sorted(range(len(spear_values)), key=lambda i: spear_values[i], reverse=True)[1] if len(spear_values) > 1 else None
            for i, m in enumerate(models):
                edrm_avg = data[m][tkey]['edrm']['avg']
                edrm_std = data[m][tkey]['edrm']['std']
                spear_avg = data[m][tkey]['spearman_correlation_coef']['avg']
                spear_std = data[m][tkey]['spearman_correlation_coef']['std']
                edrm_part = f"{round(edrm_avg,2)}$\\pm${round(edrm_std,2)}"
                spear_part = f"{round(spear_avg,2)}$\\pm${round(spear_std,2)}"
                if i == edrm_best_idx:
                    edrm_part = f"\\textbf{{{edrm_part}}}"
                elif i == edrm_second_idx:
                    edrm_part = f"\\underline{{{edrm_part}}}"
                if i == spear_best_idx:
                    spear_part = f"\\textbf{{{spear_part}}}"
                elif i == spear_second_idx:
                    spear_part = f"\\underline{{{spear_part}}}"
                metric = f"{edrm_part} / {spear_part}"
                runs_metrics.append(metric)
        elif tkey.find("frenchmedmcqa|mcqa") != -1:
            # MCQA (Hamming / Exact)
            hamming_values, exact_values = [], []
            for m in models:
                hamming_avg = data[m][tkey]['hamming_score']['avg']
                exact_avg = data[m][tkey]['exact_match']['avg']
                hamming_values.append(hamming_avg)
                exact_values.append(exact_avg)
            hamming_best_idx = sorted(range(len(hamming_values)), key=lambda i: hamming_values[i], reverse=True)[0]
            hamming_second_idx = sorted(range(len(hamming_values)), key=lambda i: hamming_values[i], reverse=True)[1] if len(hamming_values) > 1 else None
            exact_best_idx = sorted(range(len(exact_values)), key=lambda i: exact_values[i], reverse=True)[0]
            exact_second_idx = sorted(range(len(exact_values)), key=lambda i: exact_values[i], reverse=True)[1] if len(exact_values) > 1 else None
            for i, m in enumerate(models):
                hamming_avg = data[m][tkey]['hamming_score']['avg']
                hamming_std = data[m][tkey]['hamming_score']['std']
                exact_avg = data[m][tkey]['exact_match']['avg']
                exact_std = data[m][tkey]['exact_match']['std']
                hamming_part = f"{round(hamming_avg*100,2)}$\\pm${round(hamming_std*100,2)}"
                exact_part = f"{round(exact_avg*100,2)}$\\pm${round(exact_std*100,2)}"
                if i == hamming_best_idx:
                    hamming_part = f"\\textbf{{{hamming_part}}}"
                elif i == hamming_second_idx:
                    hamming_part = f"\\underline{{{hamming_part}}}"
                if i == exact_best_idx:
                    exact_part = f"\\textbf{{{exact_part}}}"
                elif i == exact_second_idx:
                    exact_part = f"\\underline{{{exact_part}}}"
                metric = f"{hamming_part} / {exact_part}"
                runs_metrics.append(metric)
        else:
            # Single metric (NER, POS, CLS)
            comparison_values = []
            for m in models:
                if tkey.find("|ner") != -1 or tkey.find("|pos") != -1:
                    f1_avg = data[m][tkey]['overall_f1']['avg']
                    f1_std = data[m][tkey]['overall_f1']['std']
                    metric = f"{round(f1_avg*100,2)}$\\pm${round(f1_std*100,2)}"
                    comparison_values.append(f1_avg)
                else:
                    wf1_avg = data[m][tkey]['weighted_f1']['avg']
                    wf1_std = data[m][tkey]['weighted_f1']['std']
                    metric = f"{round(wf1_avg*100,2)}$\\pm${round(wf1_std*100,2)}"
                    comparison_values.append(wf1_avg)
                runs_metrics.append(metric)
            sorted_idx = sorted(range(len(comparison_values)), key=lambda i: comparison_values[i], reverse=True)
            best_idx = sorted_idx[0]
            second_idx = sorted_idx[1] if len(sorted_idx) > 1 else None
            for i in range(len(runs_metrics)):
                if i == best_idx:
                    runs_metrics[i] = f"\\textbf{{{runs_metrics[i]}}}"
                elif i == second_idx:
                    runs_metrics[i] = f"\\underline{{{runs_metrics[i]}}}"

        # LaTeX line construction
        corpus_label = ""
        if not corpus_written:
            corpus_label = f"\\hline\n\n\\multirow{{{num_rows}}}{{*}}{{{corpus.replace('_','-').upper()}}}"
            corpus_written = True
        line = f"{corpus_label} & {task.upper()} & " + " & ".join(runs_metrics) + " \\\\ "
        output.append(line if corpus_label else f" & {task.upper()} & " + " & ".join(runs_metrics) + " \\\\ ")

output.append("""
\\hline
\\end{tabular}
\\caption{Performance of the baselines on the set of biomedical tasks in French. Results shown as mean$\\pm$std. Best model in bold and second best is underlined.}
\\label{table:results}
\\end{table*}
""")
print("\n".join(output))