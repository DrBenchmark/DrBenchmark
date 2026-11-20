"""Print the LaTeX table of the scores of the models on every task.

The scores are the averages over the runs, read in `stats/overall_averaged_metrics.json`,
which `ComputeMetrics.py` writes. The tasks of the table are the ones of the first model.
"""

import os
import json
import argparse

from inputs import models_to_evaluate
from paths import STATS_DIR

# Names of the models in the table
mapping = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
    "microsoft/Biomednlp-PubmedBERT-base-uncased-abstract": "OLD-PubMedBERT",
    "Dr-BERT/DrBERT-7GB": "DrBERT-FS",
    "Dr-BERT/DrBERT-4GB-CP-PubMedBERT": "DrBERT-CP",
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, nargs='+', required=False, default=None,
                        help="Models of the table, in the order of its columns (default: the models of models.txt).")
    parser.add_argument("--results", type=str, required=False, default=os.path.join(STATS_DIR, "overall_averaged_metrics.json"),
                        help="Path to the averaged scores (default: stats/overall_averaged_metrics.json).")
    return parser.parse_args()


def main():
    args = arguments()

    with open(args.results) as f:
        data = json.load(f)

    models = args.models if args.models is not None else models_to_evaluate()

    print(models)

    print(data[models[0]])
    tasks = list(data[models[0]].keys())

    output = []

    # Two columns for the dataset and the task, and one per model
    output.append("""
\\begin{table*}[t!]
\\tiny
\\centering
\\begin{tabular}{|l|l|""" + "c|" * len(models) + """}
\\hline
""")

    # A model without a name for the table keeps its own
    output.append("\\textbf{Dataset} & \\textbf{Task} & " + " & ".join(["\\textbf{" + mapping.get(m, m) + "}" for m in models]) + " \\\\ ")

    latest_corpus = None
    # latest_corpus = tasks[0].split("|")[0]

    for t in tasks:

        corpus, task, fewshot = t.split("|")

        runs_metrics = []

        for m in models:
            if m not in data or t not in data[m]:
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


if __name__ == '__main__':
    main()
