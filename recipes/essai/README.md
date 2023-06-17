# ESSAI

We manually annotated two corpora from the biomedical field. The ESSAI corpus contains clinical trial protocols in French. They were mainly obtained from the National Cancer Institute The typical protocol consists of two parts: the summary of the trial, which indicates the purpose of the trial and the methods applied; and a detailed description of the trial with the inclusion and exclusion criteria. The CAS corpus contains clinical cases published in scientific literature and training material. They are published in different journals from French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries) and are related to various medical specialties (cardiology, urology, oncology, obstetrics, pulmonology, gastro-enterology). The purpose of clinical cases is to describe clinical situations of patients. Hence, their content is close to the content of clinical narratives (description of diagnoses, treatments or procedures, evolution, family history, expected audience, etc.). In clinical cases, the negation is frequently used for describing the patient signs, symptoms, and diagnosis. Speculation is present as well but less frequently. This version only contain the annotated CAS corpus.

## Data

```plain
./data/
    ./ESSAI_neg.txt
    ./ESSAI_spec.txt
```

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB" "emea"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB" "emea"
```

## Citing the dataset

```bibtex
@article{dalloux_claveau_grabar_oliveira_moro_gumiel_carvalho_2021,
    title="{Supervised learning for the detection of negation and of its scope in French and Brazilian Portuguese biomedical corpora}",
    volume={27},
    DOI={10.1017/S1351324920000352},
    number={2},
    journal={Natural Language Engineering},
    publisher={Cambridge University Press},
    author={Dalloux, Clément and Claveau, Vincent and Grabar, Natalia and Oliveira, Lucas Emanuel Silva and Moro, Claudia Maria Cabral and Gumiel, Yohan Bonescki and Carvalho, Deborah Ribeiro},
    year={2021},
    pages={181–201}
}
```
