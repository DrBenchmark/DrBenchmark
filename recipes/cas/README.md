# CAS

We manually annotated two corpora from the biomedical field. The ESSAI corpus contains clinical trial protocols in French. They were mainly obtained from the National Cancer Institute The typical protocol consists of two parts: the summary of the trial, which indicates the purpose of the trial and the methods applied; and a detailed description of the trial with the inclusion and exclusion criteria. The CAS corpus contains clinical cases published in scientific literature and training material. They are published in different journals from French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries) and are related to various medical specialties (cardiology, urology, oncology, obstetrics, pulmonology, gastro-enterology). The purpose of clinical cases is to describe clinical situations of patients. Hence, their content is close to the content of clinical narratives (description of diagnoses, treatments or procedures, evolution, family history, expected audience, etc.). In clinical cases, the negation is frequently used for describing the patient signs, symptoms, and diagnosis. Speculation is present as well but less frequently. This version only contain the annotated CAS corpus.

## Data

```plain
./data/
    ./CAS_neg.txt/
    ./CAS_spec.txt/
```

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

## Citing the dataset

```bibtex
@inproceedings{grabar:hal-01937096,
  TITLE = "{{CAS: French Corpus with Clinical Cases}}",
  AUTHOR = {Grabar, Natalia and Claveau, Vincent and Dalloux, Cl{\'e}ment},
  URL = {https://hal.archives-ouvertes.fr/hal-01937096},
  BOOKTITLE = {{Proceedings of the 9th International Workshop on Health Text Mining and Information Analysis (LOUHI)}},
  ADDRESS = {Brussels, Belgium},
  PAGES = {1--7},
  YEAR = {2018},
  MONTH = Oct,
  PDF = {https://hal.archives-ouvertes.fr/hal-01937096/file/corpus_Louhi_2018.pdf},
  HAL_ID = {hal-01937096},
  HAL_VERSION = {v1},
}
```
