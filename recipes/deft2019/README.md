# DEFT 2019

The DEFT corpus consists of clinical cases written in French from the CAS corpus. The clinical cases are related to medical specialties (cardiology, urology, oncology, obstetrics, pneumology, gastroenterology, etc.). They have been published in various French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries, etc.).

A sub-corpus of 717 files has been compiled for the 2019 edition of the DEFT (Défi Fouille de Texte). These files were annotated manually with 4 entities: age, genre, issue, origine.

## Data

The access to the DEFT 2019 data is restricted: the corpus is not available on the HuggingFace Hub and has to be acquired first (see the [main README](../../README.md)). Once acquired, put the content of the `DEFT2021-cas-cliniques` archive directly in `./data/`:

```plain
> ./data/
    > DEFT-cas-cliniques
        > *.txt
        > *.ann
    > evaluations
        > ref-train-deft2021.txt
        > ref-test-deft2021.txt
    > README-2021.txt
    > distribution-corpus.txt
```

When `offline` is set to `False` in `config.yaml`, the corpus is read from these files by the local loading script `data_loaders_hf/DEFT2019.py`. This script only uses `distribution-corpus.txt` (assignment of each file to the training or test set of DEFT 2019) and the BRAT files of `DEFT-cas-cliniques/`. The documents are split into sentences, and the validation set is randomly taken from the training documents.

When `offline` is set to `True`, which is the default value of the repository, the recipe reads instead a dataset saved beforehand (with `save_to_disk`) in `./data/local_hf_None/`. Unlike for the other datasets, this directory is not created by `download_datasets_locally.py`, in which DEFT 2019 is commented out.

## Run Training

The launcher uses relative paths: it has to be run from the `scripts/` directory.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB"
```

Or, from the root of the repository:

```bash
python run.py --tasks deft2019-ner --models "Dr-BERT/DrBERT-7GB"
```

## Task 1 - Named-Entity Recognition

The entities are annotated in the IOB2 format.

**Entities** : `O`, `B-age`, `I-age`, `B-genre`, `I-genre`, `B-issue`, `I-issue`, `B-origine` and `I-origine`

- **Launcher**: `run.sh` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1

This task is not part of the 20 tasks evaluated in the DrBenchmark paper: no reference result is available for it.

## Citing the dataset

```bibtex
@inproceedings{Grabar-Grouin-Hamon-Claveau:DEFT:2019,
    author = "Grabar, Natalia and Grouin, Cyril and Hamon, Thierry and Claveau, Vincent",
    title = "Recherche et extraction d'information dans des cas cliniques. Pr\'esentation de la campagne d'\'evaluation DEFT 2019",
    booktitle = "Actes de la Conf\'erence sur le Traitement Automatique des Langues Naturelles (TALN)  PFIA 2019. D\'efi Fouille de Textes (atelier TALN-RECITAL)",
    month = "7",
    year = "2019",
    address = "Toulouse, France",
    publisher = "Association pour le Traitement Automatique des Langues",
    pages = "7-16",
    note = "Information Retrieval and Information Extraction from Clinical Cases",
    abstract = "Cet article pr\'esente la campagne d'\'evaluation DEFT 2019 sur l'analyse de textes cliniques r\'edig\'es en fran\c{c}ais. Le corpus se compose de cas cliniques publi\'es et discut\'es dans des articles scientifiques, et index\'es par des mots-cl\'es. Nous proposons trois t\^aches ind\'ependantes : l'indexation des cas cliniques et discussions, \'evalu\'ee prioritairement par la MAP (mean average precision), l'appariement entre cas cliniques et discussions, \'evalu\'e au moyen d'une pr\'ecision, et l'extraction d'information parmi quatre cat\'egories (\^age, genre, origine de la consultation, issue), \'evalu\'ee en termes de rappel, pr\'ecision et F-mesure. Nous pr\'esentons les r\'esultats obtenus par les participants sur chaque t\^ache.",
    keywords = "Cas clinique, fouille de texte, extraction d'information, recherche d'information,  \'evaluation.",
    url = "http://talnarchives.atala.org/ateliers/2019/DEFT/1.pdf"
}
```
