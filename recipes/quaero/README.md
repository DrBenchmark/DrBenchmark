# QUAERO French Medical

The QUAERO French Medical Corpus has been initially developed as a resource for named entity recognition and normalization [1]. It was then improved with the purpose of creating a gold standard set of normalized entities for French biomedical text, that was used in the CLEF eHealth evaluation lab [2][3].
A selection of MEDLINE titles and EMEA documents were manually annotated. The annotation process was guided by concepts in the Unified Medical Language System (UMLS):
1. Ten types of clinical entities, as defined by the following UMLS Semantic Groups (Bodenreider and McCray 2003) were annotated: Anatomy, Chemical and Drugs, Devices, Disorders, Geographic Areas, Living Beings, Objects, Phenomena, Physiology, Procedures.
2. The annotations were made in a comprehensive fashion, so that nested entities were marked, and entities could be mapped to more than one UMLS concept. In particular: (a) If a mention can refer to more than one Semantic Group, all the relevant Semantic Groups should be annotated. For instance, the mention “récidive” (recurrence) in the phrase “prévention des récidives” (recurrence prevention) should be annotated with the category “DISORDER” (CUI C2825055) and the category “PHENOMENON” (CUI C0034897); (b) If a mention can refer to more than one UMLS concept within the same Semantic Group, all the relevant concepts should be annotated. For instance, the mention “maniaques” (obsessive) in the phrase “patients maniaques” (obsessive patients) should be annotated with CUIs C0564408 and C0338831 (category “DISORDER”); (c) Entities which span overlaps with that of another entity should still be annotated. For instance, in the phrase “infarctus du myocarde” (myocardial infarction), the mention “myocarde” (myocardium) should be annotated with category “ANATOMY” (CUI C0027061) and the mention “infarctus du myocarde” should be annotated with category “DISORDER” (CUI C0027051)
The QUAERO French Medical Corpus BioC release comprises a subset of the QUAERO French Medical corpus, as follows:
Training data (BRAT version used in CLEF eHealth 2015 task 1b as training data): 
- MEDLINE_train_bioc file: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA_train_bioc file: 3 EMEA documents, segmented into 11 sub-documents, annotated with normalized entities in the BioC format 
Development data  (BRAT version used in CLEF eHealth 2015 task 1b as test data and in CLEF eHealth 2016 task 2 as development data): 
- MEDLINE_dev_bioc file: 832 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA_dev_bioc file: 3 EMEA documents, segmented into 12 sub-documents, annotated with normalized entities in the BioC format 
Test data (BRAT version used in CLEF eHealth 2016 task 2 as test data): 
- MEDLINE_test_bioc folder: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA folder_test_bioc: 4 EMEA documents, segmented into 15 sub-documents, annotated with normalized entities in the BioC format 
This release of the QUAERO French medical corpus, BioC version, comes in the BioC format, through automatic conversion from the original BRAT format obtained with the Brat2BioC tool https://bitbucket.org/nicta_biomed/brat2bioc developped by Jimeno Yepes et al.
Antonio Jimeno Yepes, Mariana Neves, Karin Verspoor 
Brat2BioC: conversion tool between brat and BioC
BioCreative IV track 1 - BioC: The BioCreative Interoperability Initiative, 2013
Please note that the original version of the QUAERO corpus distributed in the CLEF eHealth challenge 2015 and 2016 came in the BRAT stand alone format. It was distributed with the CLEF eHealth evaluation tool. This original distribution of the QUAERO French Medical corpus is available separately from https://quaerofrenchmed.limsi.fr  
All questions regarding the task or data should be addressed to aurelie.neveol@limsi.fr

## Run Training

```bash
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB" "emea"
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB" "medline"
```

## Result

```plain
                precision    recall  f1-score   support

          ANAT     0.5129    0.5930    0.5500       570
     ANAT_CHEM     0.0000    0.0000    0.0000         7
ANAT_CHEM_PROC     0.0000    0.0000    0.0000         1
     ANAT_DEVI     1.0000    0.2000    0.3333         5
     ANAT_DISO     0.5625    0.4755    0.5153       265
ANAT_DISO_PHYS     0.0000    0.0000    0.0000         1
     ANAT_LIVB     0.0000    0.0000    0.0000         2
     ANAT_OBJC     0.0000    0.0000    0.0000         1
     ANAT_PHEN     0.0000    0.0000    0.0000         2
     ANAT_PHYS     0.7333    0.6875    0.7097        16
     ANAT_PROC     0.3939    0.3095    0.3467        42
          CHEM     0.6848    0.7490    0.7155       789
     CHEM_DISO     0.7500    0.1714    0.2791        35
     CHEM_LIVB     0.0000    0.0000    0.0000         4
     CHEM_PHEN     0.0000    0.0000    0.0000         2
     CHEM_PHYS     0.0000    0.0000    0.0000         9
     CHEM_PROC     0.0000    0.0000    0.0000         8
          DEVI     0.3529    0.0811    0.1319        74
     DEVI_OBJC     0.0000    0.0000    0.0000         4
     DEVI_PHYS     0.0000    0.0000    0.0000         1
     DEVI_PROC     0.0000    0.0000    0.0000        10
          DISO     0.7517    0.7265    0.7389      2121
     DISO_LIVB     0.0909    0.0385    0.0541        52
     DISO_OBJC     0.0000    0.0000    0.0000         2
     DISO_PHEN     0.0000    0.0000    0.0000         3
     DISO_PHYS     0.1000    0.0455    0.0625        22
     DISO_PROC     0.0000    0.0000    0.0000        14
          GEOG     0.7027    0.4815    0.5714       108
     GEOG_LIVB     0.0000    0.0000    0.0000         4
          LIVB     0.6763    0.6869    0.6815       578
     LIVB_OBJC     0.0000    0.0000    0.0000         2
     LIVB_PHYS     0.0000    0.0000    0.0000         1
     LIVB_PROC     0.5000    0.2000    0.2857         5
             O     0.8675    0.9026    0.8847      9191
          OBJC     0.2692    0.1273    0.1728        55
     OBJC_PHEN     0.0000    0.0000    0.0000         1
     OBJC_PROC     0.0000    0.0000    0.0000         0
          PHEN     0.2683    0.1528    0.1947        72
     PHEN_PHYS     1.0000    1.0000    1.0000         4
     PHEN_PROC     0.0000    0.0000    0.0000         4
          PHYS     0.3427    0.2607    0.2961       234
     PHYS_PROC     0.0000    0.0000    0.0000         6
          PROC     0.6977    0.6552    0.6758      1018

      accuracy                         0.7906     15345
     macro avg     0.2618    0.1987    0.2139     15345
  weighted avg     0.7781    0.7906    0.7824     15345
```

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
