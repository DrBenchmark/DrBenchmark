# MantraGSC

We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups.

## Run Training

```bash
bash ./scripts/run.sh "Dr-BERT/DrBERT-7GB" "fr_patents"
```

## Results EMEA

```plain
              precision    recall  f1-score   support

      B-ANAT     0.4000    1.0000    0.5714         2
      B-CHEM     0.7500    0.8462    0.7952        39
      B-DISO     0.4242    0.8750    0.5714        16
      B-LIVB     1.0000    0.6471    0.7857        17
      B-OBJC     1.0000    0.3333    0.5000         3
      B-PHYS     0.8000    0.5000    0.6154         8
      B-PROC     0.7857    0.6875    0.7333        16
      I-CHEM     1.0000    0.6667    0.8000         3
      I-DISO     0.2727    0.3333    0.3000         9
      I-LIVB     0.0000    0.0000    0.0000         5
      I-PHYS     0.0000    0.0000    0.0000         1
      I-PROC     0.0000    0.0000    0.0000        10
           O     0.9450    0.9500    0.9475       380

    accuracy                         0.8684       509
   macro avg     0.5675    0.5261    0.5092       509
weighted avg     0.8652    0.8684    0.8604       509
```

## Results MEDLINE

```plain
              precision    recall  f1-score   support

      B-ANAT     0.0000    0.0000    0.0000        14
      B-CHEM     0.0000    0.0000    0.0000         2
      B-DEVI     0.0000    0.0000    0.0000         1
      B-DISO     0.4474    0.4474    0.4474        38
      B-GEOG     0.0000    0.0000    0.0000         4
      B-LIVB     1.0000    0.1887    0.3175        53
      B-OBJC     0.0000    0.0000    0.0000         5
      B-PHYS     0.0000    0.0000    0.0000        12
      B-PROC     0.5455    0.6207    0.5806        29
      I-ANAT     0.0000    0.0000    0.0000         2
      I-CHEM     0.0000    0.0000    0.0000         4
      I-DEVI     0.0000    0.0000    0.0000         1
      I-DISO     0.2941    0.4545    0.3571        11
      I-LIVB     0.0000    0.0000    0.0000         6
      I-PHYS     0.0000    0.0000    0.0000         2
           O     0.6618    0.9579    0.7828       190

    accuracy                         0.6203       374
   macro avg     0.1843    0.1668    0.1553       374
weighted avg     0.5743    0.6203    0.5436       374
```

## Results PATENTS

```plain
              precision    recall  f1-score   support

      B-ANAT     0.5000    0.5000    0.5000         2
      B-CHEM     0.5333    0.4103    0.4638        39
      B-DEVI     0.0000    0.0000    0.0000         7
      B-DISO     0.5870    0.8438    0.6923        32
      B-LIVB     0.0000    0.0000    0.0000         2
      B-OBJC     0.0000    0.0000    0.0000         2
      B-PHYS     0.0000    0.0000    0.0000         2
      B-PROC     0.0000    0.0000    0.0000         7
      I-CHEM     0.1111    0.2500    0.1538         4
      I-DEVI     0.0000    0.0000    0.0000         4
      I-DISO     0.6316    0.5714    0.6000        42
      I-PHYS     0.0000    0.0000    0.0000         2
      I-PROC     0.0000    0.0000    0.0000         5
           O     0.9326    0.9706    0.9512       613

    accuracy                         0.8702       763
   macro avg     0.2354    0.2533    0.2401       763
weighted avg     0.8378    0.8702    0.8521       763
```

## Citing the dataset

```bibtex
@article{10.1093/jamia/ocv037,
	author = {Kors, Jan A and Clematide, Simon and Akhondi,
	Saber A and van Mulligen, Erik M and Rebholz-Schuhmann, Dietrich},
	title = "{A multilingual gold-standard corpus for biomedical concept recognition: the Mantra GSC}",
	journal = {Journal of the American Medical Informatics Association},
	volume = {22},
	number = {5},
	pages = {948-956},
	year = {2015},
	month = {05},
	abstract = "{Objective To create a multilingual gold-standard corpus for biomedical concept recognition.Materials
	and methods We selected text units from different parallel corpora (Medline abstract titles, drug labels,
	biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language
	independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and
	covering a wide range of semantic groups. To reduce the annotation workload, automatically generated
	preannotations were provided. Individual annotations were automatically harmonized and then adjudicated, and
	cross-language consistency checks were carried out to arrive at the final annotations.Results The number of final
	annotations was 5530. Inter-annotator agreement scores indicate good agreement (median F-score 0.79), and are
	similar to those between individual annotators and the gold standard. The automatically generated harmonized
	annotation set for each language performed equally well as the best annotator for that language.Discussion The use
	of automatic preannotations, harmonized annotations, and parallel corpora helped to keep the manual annotation
	efforts manageable. The inter-annotator agreement scores provide a reference standard for gauging the performance
	of automatic annotation techniques.Conclusion To our knowledge, this is the first gold-standard corpus for
	biomedical concept recognition in languages other than English. Other distinguishing features are the wide variety
	of semantic groups that are being covered, and the diversity of text genres that were annotated.}",
	issn = {1067-5027},
	doi = {10.1093/jamia/ocv037},
	url = {https://doi.org/10.1093/jamia/ocv037},
	eprint = {https://academic.oup.com/jamia/article-pdf/22/5/948/34146393/ocv037.pdf},
}
```
