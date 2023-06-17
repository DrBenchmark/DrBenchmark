# MORFITT

This article presents MORFITT, the first multi-label corpus in French annotated in specialties in the medical field. MORFITT is composed of 3,624 abstracts of scientific articles from PubMed, annotated in 12 specialties for a total of 5,116 annotations. We detail the corpus, the experiments and the preliminary results obtained using a classifier based on the pre-trained language model CamemBERT. These preliminary results demonstrate the difficulty of the task, with a weighted average F-score of 61.78%.

## Run Training

```bash
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB"
```

## Citing the dataset

```bibtex
@inproceedings{labrak:hal-04125879,
  TITLE = {{MORFITT : A multi-label corpus of French scientific articles in the biomedical domain}},
  AUTHOR = {Labrak, Yanis and Rouvier, Micka{\"e}l and Dufour, Richard},
  URL = {https://hal.science/hal-04125879},
  BOOKTITLE = {{30e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles (TALN) Atelier sur l'Analyse et la Recherche de Textes Scientifiques}},
  ADDRESS = {Paris, France},
  ORGANIZATION = {{Florian Boudin}},
  YEAR = {2023},
  MONTH = Jun,
  KEYWORDS = {BERT ; RoBERTa ; Transformers ; Biomedical ; Clinical ; Topics ; multi-labels ; BERT ; RoBERTa ; Transformers ; Biom{\'e}dical ; Clinique ; Sp{\'e}cialit{\'e}s ; multi-labels},
  PDF = {https://hal.science/hal-04125879/file/_ARTS___TALN_RECITAL_2023__MORFITT__Multi_label_topic_classification_for_French_Biomedical_literature%20%285%29.pdf},
  HAL_ID = {hal-04125879},
  HAL_VERSION = {v1},
}
```
