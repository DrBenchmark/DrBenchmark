# MORFITT

This article presents MORFITT, the first multi-label corpus in French annotated in specialties in the medical field. MORFITT is composed of 3,624 abstracts of scientific articles from PubMed, annotated in 12 specialties for a total of 5,116 annotations. We detail the corpus, the experiments and the preliminary results obtained using a classifier based on the pre-trained language model CamemBERT. These preliminary results demonstrate the difficulty of the task, with a weighted average F-score of 61.78%.

## Run Training

```bash
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB"
```

## DrBERT

```plain
              precision    recall  f1-score   support

microbiology     0.6986    0.5930    0.6415        86
    etiology     0.6765    0.5180    0.5867       222
    virology     0.7826    0.5373    0.6372        67
  physiology     0.7412    0.4257    0.5408       148
  immunology     0.6500    0.5571    0.6000        70
parasitology     0.8077    0.4200    0.5526        50
    genetics     0.8400    0.6604    0.7394       159
   chemistry     0.7353    0.3846    0.5051        65
  veterinary     0.8386    0.7362    0.7841       254
     surgery     0.8611    0.7898    0.8239       157
pharmacology     0.6964    0.3786    0.4906       103
  psychology     0.8652    0.8603    0.8627       179

   micro avg     0.7861    0.6147    0.6899      1560
   macro avg     0.7661    0.5718    0.6471      1560
weighted avg     0.7785    0.6147    0.6805      1560
 samples avg     0.7520    0.6641    0.6826      1560
```

## Citing the dataset

```bibtex
TODO
```
