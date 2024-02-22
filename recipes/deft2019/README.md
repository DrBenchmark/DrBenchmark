# DEFT 2019

The DEFT corpus consists of clinical cases written in French from the CAS corpus. The clinical cases are related to medical specialties (cardiology, urology, oncology, obstetrics, pneumology, gastroenterology, etc.). They have been published in various French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries, etc.).

A sub-corpus of 717 files has been compiled for the 2019 edition of the DEFT (Défi Fouille de Texte). These files were annotated manually with 4 entities: age, genre, issue, origine.

## Data

```plain
> ./data/
    > DEFT2021-cas-cliniques
        > DEFT-cas-cliniques
            > *.txt
            > *.ann
        > evaluations
            > ref-train-deft2021.txt
            > ref-test-deft2021.txt
        > README-2021.txt
        > distribution-corpus.txt
```

## Run Training

```bash
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB"
```

## 

## Task 1 - Named-Entity Recognition

TODO

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
