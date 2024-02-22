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

## Task 1 - POS Tagging

```plain
              precision    recall  f1-score   support

         ABR     0.8692    0.8306    0.8495       248
         ADJ     0.9790    0.9767    0.9779      5540
         ADV     0.9933    0.9777    0.9854      1211
     DET:ART     0.9984    0.9993    0.9988      4298
     DET:POS     1.0000    0.9939    0.9969       164
         INT     1.0000    1.0000    1.0000         5
         KON     0.9875    0.9923    0.9899      1036
         NAM     0.9253    0.9400    0.9326      1133
         NOM     0.9848    0.9837    0.9843     10459
         NUM     0.9950    0.9950    0.9950      1797
     PRO:DEM     1.0000    1.0000    1.0000       164
     PRO:IND     0.9915    1.0000    0.9957       116
     PRO:PER     0.9916    0.9916    0.9916       713
     PRO:REL     0.9799    0.9932    0.9865       147
         PRP     0.9976    0.9986    0.9981      5021
     PRP:det     0.9965    1.0000    0.9982       845
         PUN     0.9997    0.9997    0.9997      3026
     PUN:cit     1.0000    0.9009    0.9479       111
        SENT     1.0000    1.0000    1.0000      1585
         SYM     0.8475    0.9709    0.9050       103
    VER:cond     0.8571    0.7500    0.8000         8
    VER:futu     0.9333    0.8485    0.8889        33
    VER:impf     0.9888    1.0000    0.9944      1061
    VER:infi     0.9709    0.9816    0.9762       272
    VER:pper     0.9791    0.9872    0.9832      1567
    VER:ppre     0.9872    0.9788    0.9830       236
    VER:pres     0.9916    0.9898    0.9907      1077
    VER:simp     0.9219    0.8939    0.9077        66
    VER:subi     1.0000    0.4286    0.6000         7
    VER:subp     1.0000    0.8571    0.9231         7

    accuracy                         0.9870     42056
   macro avg     0.9722    0.9420    0.9527     42056
weighted avg     0.9871    0.9870    0.9870     42056
```

## Task 2 - Negation / Speculation classification

```plain
              precision    recall  f1-score   support

    negation     0.4974    1.0000    0.6643       754
 speculation     0.0000    0.0000    0.0000       762

    accuracy                         0.4974      1516
   macro avg     0.2487    0.5000    0.3322      1516
weighted avg     0.2474    0.4974    0.3304      1516
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
