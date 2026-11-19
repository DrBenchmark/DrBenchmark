# PxCorpus

PxSLU is to the best of our knowledge, the first spoken medical drug prescriptions corpus to be distributed. It contains 4 hours of transcribed
and annotated dialogues of drug prescriptions in French acquired through an experiment with 55 participants experts and non-experts in drug prescriptions.

The automatic transcriptions were verified by human effort and aligned with semantic labels to allow training of NLP models. The data acquisition
protocol was reviewed by medical experts and permit free distribution without breach of privacy and regulation.

Overview of the Corpus

The experiment has been performed in wild conditions with naive participants and medical experts. In total, the dataset includes 1981 recordings
of 55 participants (38% non-experts, 25% doctors, 36% medical practitioners), manually transcribed and semantically annotated.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/PxCorpus](https://huggingface.co/datasets/DrBenchmark/PxCorpus). Each example is the transcription of a recording, and the same examples are used for both tasks. They are randomly split into 70% for training, 10% for validation and 20% for test:

| Train | Validation | Test |
|------:|-----------:|-----:|
| 1,386 |        198 |  397 |

The corpus is downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, it is read from `./data/local_hf_None/`: this directory is created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launchers use relative paths: they have to be run from the `scripts/` directory.

```bash
cd scripts/
bash run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash run_task_2.sh "Dr-BERT/DrBERT-7GB"
```

Or, from the root of the repository:

```bash
python run.py --tasks pxcorpus-ner pxcorpus-cls --models "Dr-BERT/DrBERT-7GB"
```

## Task 1 - Named-Entity Recognition

Each word of the transcription is classified into one of 38 classes (37 types of entities and `O`). The entities are annotated in the IOB2 format, which gives 52 labels.

**Entities** : `O`, and `B-` / `I-` for `A`, `cma_event`, `d_dos_form`, `d_dos_form_ext`, `d_dos_up`, `d_dos_val`, `dos_cond`, `dos_uf`, `dos_val`, `drug`, `dur_ut`, `dur_val`, `fasting`, `freq_days`, `freq_int_v1`, `freq_int_v1_ut`, `freq_int_v2`, `freq_int_v2_ut`, `freq_startday`, `freq_ut`, `freq_val`, `inn`, `max_unit_uf`, `max_unit_ut`, `max_unit_val`, `min_gap_ut`, `min_gap_val`, `qsp_ut`, `qsp_val`, `re_ut`, `re_val`, `rhythm_hour`, `rhythm_perday`, `rhythm_rec_ut`, `rhythm_rec_val`, `rhythm_tdte` and `roa` (not all types have an `I-` label)

- **Launcher**: `run_task_1.sh` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1
- **Reference result**: 95.88 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

## Task 2 - Intent classification

Each transcription is classified into one of the 4 following intents: `medical_prescription`, `negate`, `none` and `replace`.

- **Launcher**: `run_task_2.sh` (configuration: `yaml/cls.yaml`)
- **Metric**: weighted F1
- **Reference result**: 94.43 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

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
