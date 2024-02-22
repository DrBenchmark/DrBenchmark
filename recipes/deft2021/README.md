# DEFT 2021

The DEFT corpus consists of clinical cases written in French from the CAS corpus. The clinical cases are related to medical specialties (cardiology, urology, oncology, obstetrics, pneumology, gastroenterology, etc.). They have been published in various French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries, etc.).

A sub-corpus of 275 files has been taken from the 717 files of the 2019 edition of DEFT and has been annotated for two task: named-entity recognition and multi-label classification, corresponding to the 2020 and 2021 editions of DEFT. and 167 files compose the training set and 108 files for the test set. 


The first task is a fine-grained information extraction task based on ten categories. Although thirteen categories were annotated, only ten were used for the evaluation campaign. The annotations on the thirteen categories were, however, distributed and are included in this archive. 
Four areas are covered:
- around patients: (1) anatomy
- clinical practice: (2) examination, (3) pathology, (4) sign or symptom
- drug and surgical treatments: (5) substance, (6) dose, (7) duration, (8) frequency, (9) mode of administration, (10) treatment (surgical or medical), (11) value
- temporality: (12) date, (13) time

The annotation guide used can be consulted at: https://deft.limsi.fr/2020/guide-deft.html

For some types, attributes have been used, providing additional information that has not been used for the DEFT 2020 campaign.

The second task is the identification of patient's clinical profile based on the diseases, signs or symptoms mentioned in the clinical cases. These mentions have been annotated around the main axes of MeSH chapter C (https://mesh.inserm.fr/). While there are 26 axes in this chapter, only 23 axes were identified among the diseases, signs or symptoms mentioned in the corpus:
- C01 Infections bactériennes et mycoses
- C02 Maladies virales
- C03 Maladies parasitaires
- C04 Tumeurs
- C05 Maladies ostéomusculaires
- C06 Maladies de l'appareil digestif
- C07 Maladies du système stomatognathique
- C08 Maladies de l'appareil respiratoire
- C09 Maladies oto-rhino-laryngologiques
- C10 Maladies du système nerveux
- C11 Maladies de l'oeil
- C12 Maladies urogénitales de l'homme
- C13 Maladies de l'appareil urogénital féminin et complications de la grossesse
- C14 Maladies cardiovasculaires
- C15 Hémopathies et maladies lymphatiques
- C16 Malformations et maladies congénitales, héréditaires et néonatales
- C17 Maladies de la peau et du tissu conjonctif
- C18 Maladies métaboliques et nutritionnelles
- C19 Maladies endocriniennes
- C20 Maladies du système immunitaire
- C23 États, signes et symptômes pathologiques
- C25 Troubles dus à des produits chimiques
- C26 Plaies et blessures

The same mention may refer to one or several axes (e.g., for example, "myeloma" refers to C04 Neoplasms, C15 Hematological and lymphatic and C20 Diseases of the immune system).

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
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

## Task 1 - Named-Entity Recognition

TODO

## Task 2 - Multi-label Classification

TODO

## Citing the dataset

```bibtex
@inproceedings{cardon-etal-2020-presentation,
    title = "Pr{\'e}sentation de la campagne d{'}{\'e}valuation {DEFT} 2020 : similarit{\'e} textuelle en domaine ouvert et extraction d{'}information pr{\'e}cise dans des cas cliniques (Presentation of the {DEFT} 2020 Challenge : open domain textual similarity and precise information extraction from clinical cases )",
    author = "Cardon, R{\'e}mi  and
      Grabar, Natalia  and
      Grouin, Cyril  and
      Hamon, Thierry",
    booktitle = "Actes de la 6e conf{\'e}rence conjointe Journ{\'e}es d'{\'E}tudes sur la Parole (JEP, 33e {\'e}dition), Traitement Automatique des Langues Naturelles (TALN, 27e {\'e}dition), Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (R{\'E}CITAL, 22e {\'e}dition). Atelier D{\'E}fi Fouille de Textes",
    month = "6",
    year = "2020",
    address = "Nancy, France",
    publisher = "ATALA et AFCP",
    url = "https://aclanthology.org/2020.jeptalnrecital-deft.1",
    pages = "1--13",
    abstract = "L{'}{\'e}dition 2020 du d{\'e}fi fouille de texte (DEFT) a propos{\'e} deux t{\^a}ches autour de la similarit{\'e} textuelle et une t{\^a}che d{'}extraction d{'}information. La premi{\`e}re t{\^a}che vise {\`a} identifier le degr{\'e} de similarit{\'e} entre paires de phrases sur une {\'e}chelle de 0 (le moins similaire) {\`a} 5 (le plus similaire). Les r{\'e}sultats varient de 0,65 {\`a} 0,82 d{'}EDRM. La deuxi{\`e}me t{\^a}che consiste {\`a} d{\'e}terminer la phrase la plus proche d{'}une phrase source parmi trois phrases cibles fournies, avec des r{\'e}sultats tr{\`e}s {\'e}lev{\'e}s, variant de 0,94 {\`a} 0,99 de pr{\'e}cision. Ces deux t{\^a}ches reposent sur un corpus du domaine g{\'e}n{\'e}ral et de sant{\'e}. La troisi{\`e}me t{\^a}che propose d{'}extraire dix cat{\'e}gories d{'}informations du domaine m{\'e}dical depuis le corpus de cas cliniques de DEFT 2019. Les r{\'e}sultats varient de 0,07 {\`a} 0,66 de F-mesure globale pour la sous-t{\^a}che des pathologies et signes ou sympt{\^o}mes, et de 0,14 {\`a} 0,76 pour la sous-t{\^a}che sur huit cat{\'e}gories m{\'e}dicales. Les m{\'e}thodes utilis{\'e}es reposent sur des CRF et des r{\'e}seaux de neurones.",
    language = "French",
}
```


```bibtex
@inproceedings{grouin-etal-2021-classification,
    title = "Classification de cas cliniques et {\'e}valuation automatique de r{\'e}ponses d{'}{\'e}tudiants : pr{\'e}sentation de la campagne {DEFT} 2021 (Clinical cases classification and automatic evaluation of student answers : Presentation of the {DEFT} 2021 Challenge)",
    author = "Grouin, Cyril  and
      Grabar, Natalia  and
      Illouz, Gabriel",
    booktitle = "Actes de la 28e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles. Atelier D{\'E}fi Fouille de Textes (DEFT)",
    month = "6",
    year = "2021",
    address = "Lille, France",
    publisher = "ATALA",
    url = "https://aclanthology.org/2021.jeptalnrecital-deft.1",
    pages = "1--13",
    abstract = "Le d{\'e}fi fouille de textes (DEFT) est une campagne d{'}{\'e}valuation annuelle francophone. Nous pr{\'e}sentons les corpus et baselines {\'e}labor{\'e}es pour trois t{\^a}ches : (i) identifier le profil clinique de patients d{\'e}crits dans des cas cliniques, (ii) {\'e}valuer automatiquement les r{\'e}ponses d{'}{\'e}tudiants sur des questionnaires en ligne (Moodle) {\`a} partir de la correction de l{'}enseignant, et (iii) poursuivre une {\'e}valuation de r{\'e}ponses d{'}{\'e}tudiants {\`a} partir de r{\'e}ponses d{\'e}j{\`a} {\'e}valu{\'e}es par l{'}enseignant. Les r{\'e}sultats varient de 0,394 {\`a} 0,814 de F-mesure sur la premi{\`e}re t{\^a}che (7 {\'e}quipes), de 0,448 {\`a} 0,682 de pr{\'e}cision sur la deuxi{\`e}me (3 {\'e}quipes), et de 0,133 {\`a} 0,510 de pr{\'e}cision sur la derni{\`e}re (3 {\'e}quipes).",
    language = "French",
}
```
