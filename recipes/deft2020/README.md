# DEFT-2020

L’édition 2020 du défi fouille de texte (DEFT) a proposé deux tâches autour de la similarité textuelle et une tâche d’extraction d’information. La première tâche vise à identifier le degré de similarité entre paires de phrases sur une échelle de 0 (le moins similaire) à 5 (le plus similaire). Les résultats varient de 0,65 à 0,82 d’EDRM. La deuxième tâche consiste à déterminer la phrase la plus proche d’une phrase source parmi trois phrases cibles fournies, avec des résultats très élevés, variant de 0,94 à 0,99 de précision. Ces deux tâches reposent sur un corpus du domaine général et de santé. La troisième tâche propose d’extraire dix catégories d’informations du domaine médical depuis le corpus de cas cliniques de DEFT 2019. Les résultats varient de 0,07 à 0,66 de F-mesure globale pour la sous-tâche des pathologies et signes ou symptômes, et de 0,14 à 0,76 pour la sous-tâche sur huit catégories médicales. Les méthodes utilisées reposent sur des CRF et des réseaux de neurones.

## Data

```plain
t1-test.xml
t1-train.xml
t2-test.xml
t2-train.xml
```

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

## Task 1 - identify the degree of similarity between pairs of parallel and non-parallel sentences across multiple domains

```plain
EDRM:  0.7766715
Spearman Correlation: reference and system result are correlated
Spearman Correlation:  0.8396319104009418 ( 3.3188400671693685e-110 )
```

## Task 2 - identify possible parallel sentences for a source sentence

```plain
              precision    recall  f1-score   support

           1     0.9726    0.8875    0.9281       160
           2     0.8851    0.9747    0.9277       158
           3     0.9619    0.9528    0.9573       212

    accuracy                         0.9396       530
   macro avg     0.9399    0.9383    0.9377       530
weighted avg     0.9422    0.9396    0.9397       530
```

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
