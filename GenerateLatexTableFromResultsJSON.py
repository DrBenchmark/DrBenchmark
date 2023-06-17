import json

f = open("./stats/overall_averaged_metrics.json")
data = json.load(f)
f.close()

f_in = open("./models.txt","r")
models = ["../../../models/" + m.lower().replace("/","_") for m in f_in.read().split("\n")]
f_in.close()

print(models)

print(data[models[0]])
tasks = list(data[models[0]].keys())

output = []

output.append("""
\\begin{table*}[t!]
\\tiny
\\centering
\\begin{tabular}{|l|l|c|c|c|c|c|c|c|}
\\hline
""")

mapping = {
    "almanach_camemberta-base": "CamemBERTa",
    "camembert-base": "CamemBERT",
    "almanach_camembert-bio-base": "CamemBERT-BIO",
    "dr-bert_drbert-7gb": "DrBERT 7GB",
    "dr-bert_drbert-4gb-cp-pubmedbert": "DrBERT CP PubMedBERT",
    "microsoft_biomednlp-pubmedbert-base-uncased-abstract-fulltext": "PubMedBERT",
    "microsoft_biomednlp-pubmedbert-base-uncased-abstract": "OLD-PubMedBERT",
    "flaubert_flaubert_base_uncased": "FlauBERT",
}

output.append("\\textbf{Dataset} & \\textbf{Task} & " + " & ".join(["\\textbf{" + mapping[m.replace("../../../models/","")] + "}" for m in models]) + " \\\\ ")

latest_corpus = None
# latest_corpus = tasks[0].split("|")[0]

for t in tasks:

    corpus, task, fewshot = t.split("|")

    runs_metrics = []    

    for m in models:

        if t.find("deft2020|regr") != -1 or t.find("clister|regr") != -1:
            metric = f"{round(data[m][t]['edrm'], 2)}" + " / " + f"{round(data[m][t]['spearman_correlation_coef'], 2)}"
        
        elif t.find("frenchmedmcqa|mcqa") != -1:
            # metric = float(f"{round(data[m][t]['hamming_score'], 2)}")
            metric = f"{round(data[m][t]['hamming_score'] * 100, 2)} / {round(data[m][t]['exact_match'] * 100, 2)}"
        
        elif t.find("|ner") != -1 or t.find("|pos") != -1:
            # metric = float(f"{round(data[m][t]['overall_f1'], 2)}")
            metric = f"{round(data[m][t]['overall_f1'] * 100, 2)}"

        else:
            metric = f"{round(data[m][t]['weighted_f1'] * 100, 2)}"

        runs_metrics.append(metric)

    tt = corpus.replace('_','-').upper()

    breakline = ""

    if latest_corpus != corpus or latest_corpus == None:
        breakline = "\\hline \n\n \multirow{2}{*}{" + f"{tt}" + "}"

    # runs_metrics = ["\\textbf{" + f"{rm}" + "}" if rm == max(runs_metrics) else f"{rm}" for rm in runs_metrics]
    line = f"{breakline} & {task.upper()} & " + " & ".join(runs_metrics) + " \\\\ "
    output.append(line)

    latest_corpus = corpus

output.append("""
\\hline
\\end{tabular}
\\caption{Performance of the baselines on the set of biomedical tasks in French. Best model in bold and second is underlined.}
\\label{table:results}
\\end{table*}
""")

print("\n".join(output))
