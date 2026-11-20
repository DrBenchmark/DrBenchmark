"""Measure how much the vocabularies of the tokenizers of the models overlap.

For each pair of models, the percentage of tokens they share among the ones of both
vocabularies is written in `stats/matrix.txt`, and the matrix is drawn in
`stats/matrix_Blues.png`.
"""

import os
import argparse

import seaborn as sb
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from models_and_datasets import local_model, models_to_evaluate
from repository_paths import STATS_DIR

# Names of the models in the figure
mapping = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, nargs='+', required=False, default=None,
                        help="Name or path of the models to compare (default: the models of models.txt).")
    parser.add_argument("--offline", action="store_true",
                        help="Read the models where download_models_locally.py saves them, instead of the HuggingFace Hub.")
    return parser.parse_args()


def main():
    args = arguments()
    models = args.models if args.models is not None else models_to_evaluate()

    matrix = []

    for m1_path in models:

        line = []

        m1_tokenizer = AutoTokenizer.from_pretrained(local_model(m1_path, args.offline))
        m1_vocab = [v.replace("</w>", "").replace("▁", "").replace("##", "") for v in m1_tokenizer.get_vocab().keys()]

        for m2_path in models:

            m2_tokenizer = AutoTokenizer.from_pretrained(local_model(m2_path, args.offline))
            m2_vocab = [v.replace("</w>", "").replace("▁", "").replace("##", "") for v in m2_tokenizer.get_vocab().keys()]

            taux = len(set(m1_vocab) & set(m2_vocab)) / float(len(set(m1_vocab) | set(m2_vocab))) * 100

            print(f"{m1_path} - {m2_path} : {taux}")
            line.append(taux)

        matrix.append(line)

    print(">> Start saving scores!")

    with open(os.path.join(STATS_DIR, "matrix.txt"), "w") as f_out:
        # Write in file
        for row in matrix:
            f_out.write("\t".join([str(r) for r in row]) + "\n")

    with open(os.path.join(STATS_DIR, "matrix.txt")) as f_in:
        matrix = [[float("%.1f" % float(r)) for r in row.split("\t")] for row in f_in.read().split("\n")[:-1]]

    rotation_angle = 90

    cmap = "Blues"

    sb.set(font_scale=0.5, rc={'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff'})

    heat_map = sb.heatmap(matrix, cmap=cmap, annot=True, cbar=False, fmt='g', cbar_kws={'label': 'Percentage of tokens in commons', 'orientation': 'horizontal'})
    # A model without a name for the figure keeps its own
    heat_map.set_yticklabels([mapping.get(m, m) for m in models], rotation=0, fontsize=8)
    heat_map.set_xticklabels([mapping.get(m, m) for m in models], rotation=rotation_angle, fontsize=8)

    plt.savefig(os.path.join(STATS_DIR, f"matrix_{cmap}.png"), bbox_inches='tight')


if __name__ == '__main__':
    main()
