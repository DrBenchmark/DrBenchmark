from transformers import AutoModel, AutoTokenizer

words = [
	"asymptomatique",
	"blépharorraphie",
	"bradycardie",
	"bronchographie",
	"bronchopneumopathie",
	"dysménorrhée",
	"glaucome",
	"IRM",
	"kystectomie",
	"neuroleptique",
	"nicotine",
	"poliomyélite",
	"rhinopharyngite",
	"toxicomanie",
	"vasoconstricteur",
]

mapping = {
    "almanach/camemberta-base": "CamemBERTa",
    "camembert-base": "CamemBERT",
    "flaubert/flaubert_base_uncased": "FlauBERT",
    "Dr-BERT/DrBERT-7GB": "DrBERT",
    "Dr-BERT/DrBERT-4GB-CP-PubMedBERT": "DrBERT CP PubMedBERT",
    "almanach/camembert-bio-base": "CamemBERT-BIO",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
}

results = {}

for word in words:

    if word not in results:
        results[word] = []

    for model_name in ["almanach/camemberta-base","camembert-base","flaubert/flaubert_base_uncased","Dr-BERT/DrBERT-7GB","Dr-BERT/DrBERT-4GB-CP-PubMedBERT","almanach/camembert-bio-base","microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"]:

        print(mapping[model_name])

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized_sentence = tokenizer.tokenize(word)
        tokenized_sentence = [t.replace("##","").replace("▁","").replace("</w>","") for t in tokenized_sentence if len(t.replace("##","").replace("▁","").replace("</w>","")) > 0]
        results[word].append(tokenized_sentence)

        print(tokenized_sentence)
        print()

text = ""

text += """
\\begin{table}[]
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{cccccccc}
\\hline
\\textbf{} &
\\multicolumn{3}{c}{\\textbf{French Generalist}} &
\\multicolumn{3}{c}{\\textbf{French Biomedical}} &
\\textbf{English Biomedical} \\\\
\\hline
\\textbf{Term} &
\\textbf{CamemBERTa} &
\\textbf{CamemBERT} &
\\textbf{FlauBERT} &
\\textbf{DrBERT} &
\\textbf{DrBERT CP PubMedBERT} &
\\textbf{CamemBERT-BIO} &
\\textbf{PubMedBERT} \\\\
\\hline
"""

for w in results:
    text += "\\textit{" + w + "} & " + " & ".join(["-".join(r) if len(r) > 1 else "\\checkmark" for r in results[w]]) + " \\\\ \n"

text += """
\\hline
\\end{tabular}%
}
\\end{table}
"""

print(text)
