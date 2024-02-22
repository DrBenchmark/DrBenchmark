# E3C

E3C is a freely available multilingual corpus (English, French, Italian, Spanish, and Basque) of semantically annotated clinical narratives to allow for the linguistic analysis, benchmarking, and training of information extraction systems. It consists of two types of annotations: (i) clinical entities (e.g., pathologies), (ii) temporal information and factuality (e.g., events). Researchers can use the benchmark training and test splits of our corpus to develop and test their own models.

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB" "emea"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB" "emea"
```

## Task 1 - Clinical entities

**Entities** : `O`, `B-CLINENTITY` and `I-CLINENTITY`

```plain
              precision    recall  f1-score   support

B-CLINENTITY     0.7438    0.8275    0.7834       400
I-CLINENTITY     0.5517    0.5794    0.5652       359
           O     0.9742    0.9657    0.9699      7195

    accuracy                         0.9413      7954
   macro avg     0.7566    0.7909    0.7729      7954
weighted avg     0.9435    0.9413    0.9423      7954
```

## Task 2 - Temporal information and factuality

**Entities** : `O`, `B-EVENT`, `B-ACTOR`, `B-BODYPART`, `B-TIMEX3`, `B-RML`, `I-EVENT`, `I-ACTOR`, `I-BODYPART`, `I-TIMEX3` and `I-RML`

```plain
              precision    recall  f1-score   support

     B-ACTOR     0.0000    0.0000    0.0000         0
  B-BODYPART     0.0000    0.0000    0.0000         0
     B-EVENT     0.0000    0.0000    0.0000         0
       B-RML     0.0000    0.0000    0.0000         0
    B-TIMEX3     0.0000    0.0000    0.0000         0
     I-ACTOR     0.0000    0.0000    0.0000         0
  I-BODYPART     0.0000    0.0000    0.0000         0
     I-EVENT     0.0000    0.0000    0.0000         0
       I-RML     0.0000    0.0000    0.0000         0
    I-TIMEX3     0.0000    0.0000    0.0000         0
           O     1.0000    0.6780    0.8081      7954

    accuracy                         0.6780      7954
   macro avg     0.0909    0.0616    0.0735      7954
weighted avg     1.0000    0.6780    0.8081      7954
```
