import json


with open("./stats/overall_averaged_metrics.json") as f:
    data = json.load(f)

with open('models.txt') as f_in:
    models = [l.strip() for l in f_in if l.strip()]
    models = ["../../../models/" + m.lower().replace("/", "_") for m in models]

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

output.append("\\textbf{Dataset} & \\textbf{Task} & " + " & ".join(["\\textbf{" + mapping[m.replace("../../../models/", "")] + "}" for m in models]) + " \\\\ ")

latest_corpus = None
# latest_corpus = tasks[0].split("|")[0]

for t in tasks:

    corpus, task, fewshot = t.split("|")

    runs_metrics = []

    for m in models:

        if m not in data:
            metric = "-"
        elif t.find("deft2020|regr") != -1 or t.find("clister|regr") != -1:
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

    tt = corpus.replace('_', '-').upper()

    breakline = ""

    if latest_corpus != corpus or latest_corpus is None:
        breakline = "\\hline \n\n \\multirow{2}{*}{" + f"{tt}" + "}"

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
