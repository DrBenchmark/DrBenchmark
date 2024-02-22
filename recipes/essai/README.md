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

## Task 1 - POS Tagging

```plain
              precision    recall  f1-score   support

      @card@     0.0000    0.0000    0.0000         2
         ABR     0.7477    0.2516    0.3765       318
         ADJ     0.9385    0.9086    0.9234      5966
         ADV     0.9626    0.9678    0.9652      1676
     DET:ART     0.9968    0.9994    0.9981      8833
     DET:POS     1.0000    0.9847    0.9923       131
         FAG     0.0000    0.0000    0.0000         2
         KON     0.9770    0.9913    0.9841      2405
         NAM     0.7864    0.9186    0.8474      2397
          NN     0.9458    0.7929    0.8626       198
         NOM     0.9656    0.9652    0.9654     19092
         NUM     0.9758    0.9867    0.9812      2326
        PREF     0.0000    0.0000    0.0000         6
         PRO     0.0000    0.0000    0.0000         1
     PRO:DEM     0.9961    0.9820    0.9890       779
     PRO:IND     0.9760    0.9939    0.9849       491
     PRO:PER     0.9365    0.9530    0.9447       511
     PRO:POS     0.0000    0.0000    0.0000         2
     PRO:REL     0.9959    0.9414    0.9679       256
         PRP     0.9951    0.9982    0.9966     10420
     PRP:det     0.9996    1.0000    0.9998      2286
         PUN     0.9989    0.9997    0.9993      3761
     PUN:cit     0.9386    0.9907    0.9640       108
        SENT     0.9996    1.0000    0.9998      2755
         SYM     0.9583    0.4792    0.6389        96
    VER:cond     0.8846    0.6389    0.7419        36
    VER:futu     0.9923    0.9894    0.9909      1697
    VER:impf     0.0000    0.0000    0.0000        19
    VER:infi     0.9830    0.9991    0.9910      1157
    VER:pper     0.9407    0.9532    0.9469      2263
    VER:ppre     0.9715    0.9771    0.9743       698
    VER:pres     0.9686    0.9686    0.9686      1467
    VER:simp     0.0000    0.0000    0.0000        22
    VER:subi     0.0000    0.0000    0.0000         3
    VER:subp     0.0000    0.0000    0.0000        22

    accuracy                         0.9695     72202
   macro avg     0.7095    0.6752    0.6856     72202
weighted avg     0.9687    0.9695    0.9682     72202
```

## Task 2 - Negation / Speculation classification

```plain
              precision    recall  f1-score   support

    negation     0.5188    1.0000    0.6831      1437
 speculation     0.0000    0.0000    0.0000      1333

    accuracy                         0.5188      2770
   macro avg     0.2594    0.5000    0.3416      2770
weighted avg     0.2691    0.5188    0.3544      2770
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
