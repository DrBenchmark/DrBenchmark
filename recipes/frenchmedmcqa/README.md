# FrenchMedMCQA

This paper introduces FrenchMedMCQA, the first publicly available Multiple-Choice Question Answering (MCQA) dataset in French for medical domain. It is composed of 3,105 questions taken from real exams of the French medical specialization diploma in pharmacy, mixing single and multiple answers. Each instance of the dataset contains an identifier, a question, five possible answers and their manual correction(s). We also propose first baseline models to automatically process this MCQA task in order to report on the current performances and to highlight the difficulty of the task. A detailed analysis of the results showed that it is necessary to have representations adapted to the medical domain or to the MCQA task: in our case, English specialized models yielded better results than generic French ones, even though FrenchMedMCQA is in French. Corpus, models and tools are available online.

## Run Training

```bash
bash run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash run_task_2.sh "Dr-BERT/DrBERT-7GB"
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
