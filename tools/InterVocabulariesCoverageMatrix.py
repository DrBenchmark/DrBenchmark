# python InterVocabulariesCoverageMatrix.py

import seaborn as sb
import matplotlib.pyplot as plt

from transformers import AutoTokenizer


with open('models.txt') as f_in:
    models = [l.strip() for l in f_in if l.strip()]

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

matrix = []

for m1_path in models:

    line = []

    m1_tokenizer = AutoTokenizer.from_pretrained(m1_path)
    m1_vocab = [v.replace("</w>", "").replace("▁", "").replace("##", "") for v in m1_tokenizer.get_vocab().keys()]

    for m2_path in models:

        m2_tokenizer = AutoTokenizer.from_pretrained(m2_path)
        m2_vocab = [v.replace("</w>", "").replace("▁", "").replace("##", "") for v in m2_tokenizer.get_vocab().keys()]

        taux = len(set(m1_vocab) & set(m2_vocab)) / float(len(set(m1_vocab) | set(m2_vocab))) * 100

        print(f"{m1_path} - {m2_path} : {taux}")
        line.append(taux)

    matrix.append(line)

print(">> Start saving scores!")

with open("./stats/matrix.txt", "w") as f_out:
    # Write in file
    for row in matrix:
        f_out.write("\t".join([str(r) for r in row]) + "\n")

with open("./stats/matrix.txt") as f_in:
    matrix = [[float("%.1f" % float(r)) for r in row.split("\t")] for row in f_in.read().split("\n")[:-1]]

# mask = np.zeros_like(matrix)

# for i in range(len(mask)):
#     for j in range(i+1, len(mask)):
#         mask[i,j] = 1

# mask = np.array(mask, dtype=np.bool)

rotation_angle = 90

cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)

cmap = "Blues"

sb.set(font_scale=0.5, rc={'axes.facecolor': '#ffffff', 'figure.facecolor': '#ffffff'})

heat_map = sb.heatmap(matrix, cmap=cmap, annot=True, cbar=False, fmt='g', cbar_kws={'label': 'Percentage of tokens in commons', 'orientation': 'horizontal'})
# heat_map = sb.heatmap(matrix, mask=mask, cmap=cmap, annot=True, cbar=False, fmt='g', cbar_kws={'label': 'Percentage of tokens in commons', 'orientation': 'horizontal'})
heat_map.set_yticklabels([mapping[m] for m in models], rotation=0, fontsize=8)
heat_map.set_xticklabels([mapping[m] for m in models], rotation=rotation_angle, fontsize=8)

plt.savefig(f"./stats/matrix_{cmap}.png", bbox_inches='tight')
