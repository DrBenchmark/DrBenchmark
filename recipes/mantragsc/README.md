# MantraGSC

We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups.

Only the French part of the corpus is used in DrBenchmark.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/MANTRAGSC](https://huggingface.co/datasets/DrBenchmark/MANTRAGSC). The recipe uses the three French subsets, one per source. Each subset is randomly split into 70% for training, 10% for validation and 20% for test:

| Subset       | Source                    | Train | Validation | Test |
|:-------------|:--------------------------|------:|-----------:|-----:|
| `fr_emea`    | Drug labels (EMEA)        |    70 |         10 |   20 |
| `fr_medline` | Medline abstract titles   |    70 |         10 |   20 |
| `fr_patents` | Biomedical patent claims  |    35 |          5 |   10 |

The subsets are downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, they are read from `./data/local_hf_<subset>/`: these directories are created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launcher can be run from any directory: it finds its files from its own location. Its second argument is the subset.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB" "fr_emea"
bash run.sh "Dr-BERT/DrBERT-7GB" "fr_medline"
bash run.sh "Dr-BERT/DrBERT-7GB" "fr_patents"
```

Or, from the root of the repository:

```bash
python run.py --tasks mantragsc-ner_emea mantragsc-ner_medline mantragsc-ner_patents --models "Dr-BERT/DrBERT-7GB"
```

## Named-Entity Recognition

The entities are annotated in the IOB2 format with UMLS Semantic Groups.

**Entities** : `O`, and `B-` / `I-` for `ANAT`, `CHEM`, `DEVI`, `DISO`, `LIVB`, `OBJC`, `PHEN`, `PHYS` and `PROC`. The subset `fr_medline` also contains `GEOG`.

- **Launcher**: `run.sh` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1
- **Reference results** for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938)):

| Subset       | SeqEval F1 |
|:-------------|-----------:|
| `fr_emea`    |      66.23 |
| `fr_medline` |      42.38 |
| `fr_patents` |      57.34 |

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
