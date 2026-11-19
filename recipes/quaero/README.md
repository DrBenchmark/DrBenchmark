# QUAERO French Medical

The QUAERO French Medical Corpus has been initially developed as a resource for named entity recognition and normalization [1]. It was then improved with the purpose of creating a gold standard set of normalized entities for French biomedical text, that was used in the CLEF eHealth evaluation lab [2][3].

A selection of MEDLINE titles and EMEA documents were manually annotated. The annotation process was guided by concepts in the Unified Medical Language System (UMLS):

1. Ten types of clinical entities, as defined by the following UMLS Semantic Groups (Bodenreider and McCray 2003) were annotated: Anatomy, Chemical and Drugs, Devices, Disorders, Geographic Areas, Living Beings, Objects, Phenomena, Physiology, Procedures.
2. The annotations were made in a comprehensive fashion, so that nested entities were marked, and entities could be mapped to more than one UMLS concept. In particular: (a) If a mention can refer to more than one Semantic Group, all the relevant Semantic Groups should be annotated. For instance, the mention “récidive” (recurrence) in the phrase “prévention des récidives” (recurrence prevention) should be annotated with the category “DISORDER” (CUI C2825055) and the category “PHENOMENON” (CUI C0034897); (b) If a mention can refer to more than one UMLS concept within the same Semantic Group, all the relevant concepts should be annotated. For instance, the mention “maniaques” (obsessive) in the phrase “patients maniaques” (obsessive patients) should be annotated with CUIs C0564408 and C0338831 (category “DISORDER”); (c) Entities which span overlaps with that of another entity should still be annotated. For instance, in the phrase “infarctus du myocarde” (myocardial infarction), the mention “myocarde” (myocardium) should be annotated with category “ANATOMY” (CUI C0027061) and the mention “infarctus du myocarde” should be annotated with category “DISORDER” (CUI C0027051)

The QUAERO French Medical Corpus BioC release comprises a subset of the QUAERO French Medical corpus, as follows:

Training data (BRAT version used in CLEF eHealth 2015 task 1b as training data):

- MEDLINE_train_bioc file: 833 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA_train_bioc file: 3 EMEA documents, segmented into 11 sub-documents, annotated with normalized entities in the BioC format

Development data (BRAT version used in CLEF eHealth 2015 task 1b as test data and in CLEF eHealth 2016 task 2 as development data):

- MEDLINE_dev_bioc file: 832 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA_dev_bioc file: 3 EMEA documents, segmented into 12 sub-documents, annotated with normalized entities in the BioC format

Test data (BRAT version used in CLEF eHealth 2016 task 2 as test data):

- MEDLINE_test_bioc folder: 833 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA folder_test_bioc: 4 EMEA documents, segmented into 15 sub-documents, annotated with normalized entities in the BioC format

This release of the QUAERO French medical corpus, BioC version, comes in the BioC format, through automatic conversion from the original BRAT format obtained with the Brat2BioC tool https://bitbucket.org/nicta_biomed/brat2bioc developped by Jimeno Yepes et al.

Antonio Jimeno Yepes, Mariana Neves, Karin Verspoor.
Brat2BioC: conversion tool between brat and BioC.
BioCreative IV track 1 - BioC: The BioCreative Interoperability Initiative, 2013

Please note that the original version of the QUAERO corpus distributed in the CLEF eHealth challenge 2015 and 2016 came in the BRAT stand alone format. It was distributed with the CLEF eHealth evaluation tool. This original distribution of the QUAERO French Medical corpus is available separately from https://web.archive.org/web/20240624042319/https://quaerofrenchmed.limsi.fr/

All questions regarding the task or data should be addressed to aurelie.neveol@limsi.fr

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/QUAERO](https://huggingface.co/datasets/DrBenchmark/QUAERO). It provides one subset per text genre, with the training, development and test sets of the corpus:

| Subset    | Train | Validation | Test |
|:----------|------:|-----------:|-----:|
| `emea`    |   482 |        437 |  427 |
| `medline` |   833 |        832 |  833 |

This version differs from the original corpus on two points:

- Nested entities are not kept: when several annotations overlap, only the longest one is used.
- The EMEA documents are split into sentences, as some of them are longer than the maximum input length of the models. The MEDLINE titles are left as they are. The table above therefore counts sentences for `emea` and titles for `medline`.

The subsets are downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, they are read from `./data/local_hf_<subset>/`: these directories are created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launcher uses relative paths: it has to be run from the `scripts/` directory. Its second argument is the subset.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB" "emea"
bash run.sh "Dr-BERT/DrBERT-7GB" "medline"
```

Or, from the root of the repository:

```bash
python run.py --tasks quaero-ner_emea quaero-ner_medline --models "Dr-BERT/DrBERT-7GB"
```

## Named-Entity Recognition

The entities are annotated in the IOB2 format with the ten UMLS Semantic Groups, which gives 21 labels.

**Entities** : `O`, and `B-` / `I-` for `ANAT`, `CHEM`, `DEVI`, `DISO`, `GEOG`, `LIVB`, `OBJC`, `PHEN`, `PHYS` and `PROC`

- **Launcher**: `run.sh` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1
- **Reference results** for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938)):

| Subset    | SeqEval F1 |
|:----------|-----------:|
| `emea`    |      64.11 |
| `medline` |      55.82 |

## Citing the dataset

```bibtex
@InProceedings{neveol14quaero, 
  author = {Névéol, Aurélie and Grouin, Cyril and Leixa, Jeremy 
            and Rosset, Sophie and Zweigenbaum, Pierre},
  title = {The {QUAERO} {French} Medical Corpus: A Ressource for
           Medical Entity Recognition and Normalization}, 
  OPTbooktitle = {Proceedings of the Fourth Workshop on Building 
                 and Evaluating Ressources for Health and Biomedical 
                 Text Processing}, 
  booktitle = {Proc of BioTextMining Work}, 
  OPTseries = {BioTxtM 2014}, 
  year = {2014}, 
  pages = {24--30}, 
}
```
