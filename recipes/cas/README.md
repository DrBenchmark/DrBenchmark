# CAS

We manually annotated two corpora from the biomedical field. The ESSAI corpus contains clinical trial protocols in French. They were mainly obtained from the National Cancer Institute. The typical protocol consists of two parts: the summary of the trial, which indicates the purpose of the trial and the methods applied; and a detailed description of the trial with the inclusion and exclusion criteria. The CAS corpus contains clinical cases published in scientific literature and training material. They are published in different journals from French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries) and are related to various medical specialties (cardiology, urology, oncology, obstetrics, pulmonology, gastro-enterology). The purpose of clinical cases is to describe clinical situations of patients. Hence, their content is close to the content of clinical narratives (description of diagnoses, treatments or procedures, evolution, family history, expected audience, etc.). In clinical cases, the negation is frequently used for describing the patient signs, symptoms, and diagnosis. Speculation is present as well but less frequently. This version only contains the annotated CAS corpus.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/CAS](https://huggingface.co/datasets/DrBenchmark/CAS). It provides one subset per task. Each example is a sentence, and the sentences are randomly split into 70% for training, 10% for validation and 20% for test:

| Subset     | Task                                           | Train | Validation | Test |
|:-----------|:-----------------------------------------------|------:|-----------:|-----:|
| `pos`      | Task 1 - POS Tagging                           | 2,653 |        379 |  758 |
| `cls`      | Task 2 - Negation / Speculation classification | 2,653 |        379 |  758 |
| `ner_neg`  | Task 3 - Negation scope detection              | 2,653 |        379 |  758 |
| `ner_spec` | Task 4 - Speculation scope detection           | 2,653 |        379 |  758 |

The subsets are downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, they are read from `./data/local_hf_<subset>/`: these directories are created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launchers can be run from any directory: they find their files from their own location.

```bash
cd scripts/
bash run_task_1.sh "Dr-BERT/DrBERT-7GB"
bash run_task_2.sh "Dr-BERT/DrBERT-7GB"
bash run_task_3.sh "Dr-BERT/DrBERT-7GB"
bash run_task_4.sh "Dr-BERT/DrBERT-7GB"
```

Or, from the root of the repository:

```bash
python run.py --tasks cas-pos cas-cls cas-ner_neg cas-ner_spec --models "Dr-BERT/DrBERT-7GB"
```

## Task 1 - POS Tagging

Each token is annotated with one of the 31 following part-of-speech tags: `ABR`, `ADJ`, `ADV`, `DET:ART`, `DET:POS`, `INT`, `KON`, `NAM`, `NOM`, `NUM`, `PRO:DEM`, `PRO:IND`, `PRO:PER`, `PRO:REL`, `PRP`, `PRP:det`, `PUN`, `PUN:cit`, `SENT`, `SYM`, `VER:con`, `VER:cond`, `VER:futu`, `VER:impf`, `VER:infi`, `VER:pper`, `VER:ppre`, `VER:pres`, `VER:simp`, `VER:subi` and `VER:subp`. In the dataset, the tags are stored in the IOB2 format, i.e. prefixed with `B-` (e.g. `B-NOM`).

- **Launcher**: `run_task_1.sh` (configuration: `yaml/pos.yaml`)
- **Metric**: SeqEval F1
- **Reference result**: 96.93 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

## Task 2 - Negation / Speculation classification

Each sentence is classified into one of 4 classes, according to the phenomena it contains: `negation`, `speculation`, `negation_speculation` (both) or `neutral` (none).

- **Launcher**: `run_task_2.sh` (configuration: `yaml/cls.yaml`)
- **Metric**: weighted F1

## Task 3 - Negation scope detection

Token classification task, which consists in finding the tokens that are in the scope of a negation.

**Entities** : `O`, `B_scope_neg` and `I_scope_neg`

- **Launcher**: `run_task_3.sh` (configuration: `yaml/ner_neg.yaml`)
- **Metric**: SeqEval F1

## Task 4 - Speculation scope detection

Token classification task, which consists in finding the tokens that are in the scope of a speculation.

**Entities** : `O`, `B_xcope_inc` and `I_xcope_inc`

- **Launcher**: `run_task_4.sh` (configuration: `yaml/ner_spec.yaml`, same Python script as Task 3)
- **Metric**: SeqEval F1

The tasks 2, 3 and 4 are not part of the 20 tasks evaluated in the DrBenchmark paper: no reference result is available for them.

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
