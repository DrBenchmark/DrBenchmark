# E3C

E3C is a freely available multilingual corpus (English, French, Italian, Spanish, and Basque) of semantically annotated clinical narratives to allow for the linguistic analysis, benchmarking, and training of information extraction systems. It consists of two types of annotations: (i) clinical entities (e.g., pathologies), (ii) temporal information and factuality (e.g., events). Researchers can use the benchmark training and test splits of our corpus to develop and test their own models.

Only the French part of the corpus is used in DrBenchmark.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/E3C](https://huggingface.co/datasets/DrBenchmark/E3C). The recipe uses the two French subsets, one per type of annotation. Each example is a sentence:

| Subset            | Task                                         | Train | Validation |  Test |
|:------------------|:---------------------------------------------|------:|-----------:|------:|
| `French_clinical` | Task 1 - Clinical entities                   | 2,087 |        302 | 1,109 |
| `French_temporal` | Task 2 - Temporal information and factuality |   776 |        111 |   222 |

The original corpus does not come with predefined subsets. For `French_clinical`, the training and validation sets are taken from the layer 2 of the corpus and the test set is the whole layer 1. For `French_temporal`, the layer 1 is randomly split into 70% for training, 10% for validation and 20% for test.

The subsets are downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, they are read from `./data/local_hf_<subset>/`: these directories are created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launcher can be run from any directory: it finds its files from its own location. Its second argument is the subset.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB" "French_clinical"
bash run.sh "Dr-BERT/DrBERT-7GB" "French_temporal"
```

Or, from the root of the repository:

```bash
python run.py --tasks e3c-ner_clinical e3c-ner_temporal --models "Dr-BERT/DrBERT-7GB"
```

## Task 1 - Clinical entities

**Entities** : `O`, `B-CLINENTITY` and `I-CLINENTITY`

- **Launcher**: `run.sh` with the subset `French_clinical` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1
- **Reference result**: 54.45 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

## Task 2 - Temporal information and factuality

**Entities** : `O`, `B-EVENT`, `B-ACTOR`, `B-BODYPART`, `B-TIMEX3`, `B-RML`, `I-EVENT`, `I-ACTOR`, `I-BODYPART`, `I-TIMEX3` and `I-RML`

- **Launcher**: `run.sh` with the subset `French_temporal` (configuration: `yaml/ner.yaml`)
- **Metric**: SeqEval F1
- **Reference result**: 81.48 for `Dr-BERT/DrBERT-7GB` (average of 4 runs, reported in Table 6 of the [DrBenchmark paper](https://hal.science/hal-04470938))

## Citing the dataset

```bibtex
@article{Magnini2020TheEP,
  title={The E3C Project: Collection and Annotation of a Multilingual Corpus of Clinical Cases},
  author={Bernardo Magnini and Bego{\~n}a Altuna and Alberto Lavelli and Manuela Speranza and Roberto Zanoli},
  journal={Proceedings of the Seventh Italian Conference on Computational Linguistics CLiC-it 2020},
  year={2020}
}
```
