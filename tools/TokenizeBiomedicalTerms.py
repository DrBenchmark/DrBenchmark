"""Show how the tokenizer of each model splits some biomedical terms.

The sub-word tokens of each term are printed for each model, then a LaTeX table with one
row per term and one column per model: a term that a tokenizer keeps whole has a check
mark.
"""

import argparse

from transformers import AutoTokenizer

from models_and_datasets import local_model

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

# The models of the table, when none is given
default_models = [
    "almanach/camemberta-base", "camembert-base", "flaubert/flaubert_base_uncased", "Dr-BERT/DrBERT-7GB",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "xlm-roberta-base",
]

# Names of the models in the table
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
    parser.add_argument("--models", type=str, nargs='+', required=False, default=default_models,
                        help="Name or path of the models of the table (default: %(default)s).")
    parser.add_argument("--offline", action="store_true",
                        help="Read the models where download_models_locally.py saves them, instead of the HuggingFace Hub.")
    return parser.parse_args()


def main():
    args = arguments()
    models = args.models
    # A model without a name for the table keeps its own
    names = [mapping.get(model_name, model_name) for model_name in models]

    tokenizers = {}
    results = {}

    for word in words:

        if word not in results:
            results[word] = []

        for model_name, name in zip(models, names):

            print(name)

            if model_name not in tokenizers:
                tokenizers[model_name] = AutoTokenizer.from_pretrained(local_model(model_name, args.offline))
            tokenizer = tokenizers[model_name]

            tokenized_sentence = tokenizer.tokenize(word)
            tokenized_sentence = [t.replace("##", "").replace("▁", "").replace("</w>", "") for t in tokenized_sentence if len(t.replace("##", "").replace("▁", "").replace("</w>", "")) > 0]
            results[word].append(tokenized_sentence)

            print(tokenized_sentence)
            print()

    text = ""

    # One column for the terms, and one per model
    text += """
\\begin{table}[]
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{""" + "c" * (len(models) + 1) + """}
\\hline
\\textbf{Term} &
"""
    text += " &\n".join(["\\textbf{" + name + "}" for name in names]) + " \\\\\n\\hline\n"

    for w in results:
        text += "\\textit{" + w + "} & " + " & ".join(["-".join(r) if len(r) > 1 else "\\checkmark" for r in results[w]]) + " \\\\ \n"

    text += """
\\hline
\\end{tabular}%
}
\\end{table}
"""

    print(text)


if __name__ == '__main__':
    main()
