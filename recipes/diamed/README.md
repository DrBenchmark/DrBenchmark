# DiaMED

DiaMED is a corpus of 739 clinical cases written in French, which has been created for DrBenchmark. The clinical cases were collected from an open-access journal, The Pan African Medical Journal. Each of them was manually assigned by several annotators, including a medical expert, to one of the 22 chapters of the International Classification of Diseases, 10th Revision (ICD-10). A chapter gives a general description of the type of disease or injury.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/DiaMED](https://huggingface.co/datasets/DrBenchmark/DiaMED).

| Train | Validation | Test |
|------:|-----------:|-----:|
|   509 |         76 |  154 |

The dataset also contains the figures of the clinical cases, which are not used by this recipe: about 730 MB have to be downloaded.

The corpus is downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, it is read from `./data/local_hf_None/`: this directory is created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launcher uses relative paths: it has to be run from the `scripts/` directory.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB"
```

Or, from the root of the repository:

```bash
python run.py --tasks diamed-cls --models "Dr-BERT/DrBERT-7GB"
```

## Multi-class Classification

Each clinical case (field `clinical_case`) is classified into one of the 22 following ICD-10 chapters (field `icd-10`):

- A00-B99 Certain infectious and parasitic diseases
- C00-D49 Neoplasms
- D50-D89 Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
- E00-E89 Endocrine, nutritional and metabolic diseases
- F01-F99 Mental, Behavioral and Neurodevelopmental disorders
- G00-G99 Diseases of the nervous system
- H00-H59 Diseases of the eye and adnexa
- H60-H95 Diseases of the ear and mastoid process
- I00-I99 Diseases of the circulatory system
- J00-J99 Diseases of the respiratory system
- K00-K95 Diseases of the digestive system
- L00-L99 Diseases of the skin and subcutaneous tissue
- M00-M99 Diseases of the musculoskeletal system and connective tissue
- N00-N99 Diseases of the genitourinary system
- O00-O9A Pregnancy, childbirth and the puerperium
- P00-P96 Certain conditions originating in the perinatal period
- Q00-Q99 Congenital malformations, deformations and chromosomal abnormalities
- R00-R99 Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
- S00-T88 Injury, poisoning and certain other consequences of external causes
- U00-U85 Codes for special purposes
- V00-Y99 External causes of morbidity
- Z00-Z99 Factors influencing health status and contact with health services

The text of the clinical case is lowercased and its stop words are removed before being given to the model.

- **Launcher**: `run.sh` (configuration: `yaml/cls.yaml`)
- **Metric**: weighted F1
- **Reference result**: 60.45 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

## Citing the dataset

```bibtex
@inproceedings{labrak:hal-04470938,
  title = {{DrBenchmark: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain}},
  author = {Labrak, Yanis and Bazoge, Adrien and El Khettari, Oumaima and Rouvier, Micka{\"e}l and Constant Dit Beaufils, Pac{\^o}me and Grabar, Natalia and Daille, B{\'e}atrice and Quiniou, Solen and Morin, Emmanuel and Gourraud, Pierre-antoine and Dufour, Richard},
  url = {https://hal.science/hal-04470938},
  booktitle = {{Fourteenth Language Resources and Evaluation Conference (LREC-COLING 2024)}},
  address = {Torino, Italy},
  organization = {{Nicoletta Calzolari and Min-Yen Kan}},
  year = {2024},
  month = May,
  keywords = {NLP evaluation ; Benchmarking ; Medical domain ; French language ; Transformers},
  pdf = {https://hal.science/hal-04470938/file/DrBenchmark.pdf},
  hal_id = {hal-04470938},
  hal_version = {v1},
}
```
