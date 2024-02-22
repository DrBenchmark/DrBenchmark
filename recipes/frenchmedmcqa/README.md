# FrenchMedMCQA

This paper introduces FrenchMedMCQA, the first publicly available Multiple-Choice Question Answering (MCQA) dataset in French for medical domain. It is composed of 3,105 questions taken from real exams of the French medical specialization diploma in pharmacy, mixing single and multiple answers. Each instance of the dataset contains an identifier, a question, five possible answers and their manual correction(s). We also propose first baseline models to automatically process this MCQA task in order to report on the current performances and to highlight the difficulty of the task. A detailed analysis of the results showed that it is necessary to have representations adapted to the medical domain or to the MCQA task: in our case, English specialized models yielded better results than generic French ones, even though FrenchMedMCQA is in French. Corpus, models and tools are available online.

## Run Training

```bash
bash run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

## Task 1 - Automatically identify the set of correct answers among the five proposed for a given question. (Multiple-choice question answering - MCQA)

```plain
Hamming Score: 0.3370846730975347
EMR: 0.14790996784565916
```

## Task 2 - Identify the number of answers (between 1 and 5) supposedly correct for a given question. (Introduced during DEFT 2023)

```plain
              precision    recall  f1-score   support

           1     0.9903    0.9502    0.9698       321
           2     0.5000    0.0825    0.1416        97
           3     0.4514    0.9220    0.6061       141
           4     0.3000    0.0536    0.0909        56
           5     0.0000    0.0000    0.0000         7

    accuracy                         0.7170       622
   macro avg     0.4483    0.4016    0.3617       622
weighted avg     0.7184    0.7170    0.6681       622
```

## Citing the dataset

```bibtex
@article{labrak2023frenchmedmcqa,
  title={FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain},
  author={Labrak, Yanis and Bazoge, Adrien and Dufour, Richard and Rouvier, Mickael and Morin, Emmanuel and Daille, B{\'e}atrice and Gourraud, Pierre-Antoine},
  journal={arXiv preprint arXiv:2304.04280},
  year={2023}
}
```
