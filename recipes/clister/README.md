# CLISTER

Le TAL repose sur la disponibilité de corpus annotés pour l’entraînement et l’évaluation de modèles. Il existe très peu de ressources pour la similarité sémantique dans le domaine clinique en français. Dans cette étude, nous proposons une définition de la similarité guidée par l’analyse clinique et l’appliquons au développement d’un nouveau corpus partagé de 1 000 paires de phrases annotées manuellement en scores de similarité. Nous évaluons ensuite le corpus par des expériences de mesure automatique de similarité. Nous montrons ainsi qu’un modèle de plongements de phrases peut capturer la similarité avec des performances à l’état de l’art sur le corpus DEFT STS (Spearman=0,8343). Nous montrons également que le contenu du corpus CLISTER est complémentaire de celui de DEFT STS.

## Data

The corpus is hosted on the HuggingFace Hub: [DrBenchmark/CLISTER](https://huggingface.co/datasets/DrBenchmark/CLISTER) (subset `source`). Each example is a pair of sentences (`text_1` and `text_2`) with its similarity score (`label`, between 0 and 5). The validation set is randomly taken from the original training set:

| Subset   | Train | Validation | Test |
|:---------|------:|-----------:|-----:|
| `source` |   499 |        101 |  400 |

The subset is downloaded automatically when `offline` is set to `False` in `config.yaml`. When it is set to `True`, which is the default value of the repository, it is read from `./data/local_hf_source/`: this directory is created by `python download_datasets_locally.py`, run from the root of the repository.

## Run Training

The launcher can be run from any directory: it finds its files from its own location.

```bash
cd scripts/
bash run.sh "Dr-BERT/DrBERT-7GB"
```

Or, from the root of the repository:

```bash
python run.py --tasks clister-regr --models "Dr-BERT/DrBERT-7GB"
```

## Result

- **Launcher**: `run.sh` (configuration: `yaml/regr.yaml`)
- **Metrics**: EDRM and Spearman correlation

```plain
EDRM:  0.6228134
Spearman Correlation: reference and system result are correlated
Spearman Correlation:  0.5433963048273215 ( 4.163071421358024e-32 )
```

## Citing the dataset

```bibtex
@inproceedings{hiebel-etal-2022-clister,
    title = "{CLISTER} : Un corpus pour la similarit{\'e} s{\'e}mantique textuelle dans des cas cliniques en fran{\c{c}}ais ({CLISTER} : A Corpus for Semantic Textual Similarity in {F}rench Clinical Narratives)",
    author = {Hiebel, Nicolas  and
      Fort, Kar{\"e}n  and
      N{\'e}v{\'e}ol, Aur{\'e}lie  and
      Ferret, Olivier},
    booktitle = "Actes de la 29e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles. Volume 1 : conf{\'e}rence principale",
    month = "6",
    year = "2022",
    address = "Avignon, France",
    publisher = "ATALA",
    url = "https://aclanthology.org/2022.jeptalnrecital-taln.28",
    pages = "287--296",
    abstract = "Le TAL repose sur la disponibilit{\'e} de corpus annot{\'e}s pour l{'}entra{\^\i}nement et l{'}{\'e}valuation de mod{\`e}les. Il existe tr{\`e}s peu de ressources pour la similarit{\'e} s{\'e}mantique dans le domaine clinique en fran{\c{c}}ais. Dans cette {\'e}tude, nous proposons une d{\'e}finition de la similarit{\'e} guid{\'e}e par l{'}analyse clinique et l{'}appliquons au d{\'e}veloppement d{'}un nouveau corpus partag{\'e} de 1 000 paires de phrases annot{\'e}es manuellement en scores de similarit{\'e}. Nous {\'e}valuons ensuite le corpus par des exp{\'e}riences de mesure automatique de similarit{\'e}. Nous montrons ainsi qu{'}un mod{\`e}le de plongements de phrases peut capturer la similarit{\'e} avec des performances {\`a} l{'}{\'e}tat de l{'}art sur le corpus DEFT STS (Spearman=0,8343). Nous montrons {\'e}galement que le contenu du corpus CLISTER est compl{\'e}mentaire de celui de DEFT STS.",
    language = "French",
}
```
