# PxCorpus

PxSLU is to the best of our knowledge, the first spoken medical drug prescriptions corpus to be distributed. It contains 4 hours of transcribed
and annotated dialogues of drug prescriptions in French acquired through an experiment with 55 participants experts and non-experts in drug prescriptions.

The automatic transcriptions were verified by human effort and aligned with semantic labels to allow training of NLP models. The data acquisition
protocol was reviewed by medical experts and permit free distribution without breach of privacy and regulation.

Overview of the Corpus

The experiment has been performed in wild conditions with naive participants and medical experts. In total, the dataset includes 1981 recordings
of 55 participants (38% non-experts, 25% doctors, 36% medical practitioners), manually transcribed and semantically annotated.

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

## Citing the dataset

```bibtex
@InProceedings{Kocabiyikoglu2022,
  author =     "Alican Kocabiyikoglu and Fran{\c c}ois Portet and Prudence Gibert and Hervé Blanchon and Jean-Marc Babouchkine and Gaëtan Gavazzi",
  title =     "A Spoken Drug Prescription Dataset in French for Spoken Language Understanding",
  booktitle =     "13th Language Resources and Evaluation Conference (LREC 2022)",
  year =     "2022",
  location =     "Marseille, France"
}
```
