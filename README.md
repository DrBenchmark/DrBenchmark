<p align="center">
  <img src="./assets/logo.svg" alt="drawing" width="250"/>
</p>

# DrBenchmark: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain

## Introduction

TODO

## Steps

1. Add the files for the restricted corpus (CAS, ESSAI and CLISTER) in the `./recipes/<corpus_name>/data/` folder.
2. Setup and activate the conda environement
3. (Optionnal) In case you are running the benchmark on a offline machine / cluster, please build and save locally each datasets by simply using `python ./download_datasets_locally.py`, download all the models locally by using `python ./download_models_locally.py` and finally, set the value of `offline` to `True` in the `./config.yaml` file.
4. Run the benchmark.

## Anaconda setup

```bash
conda create --name DrBenchmark python=3.9 -y
conda activate DrBenchmark
```

More information on managing environments with Anaconda can be found in [the conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

Once you have created your Python environment (Python 3.9+) you can simply type:

```bash
git clone https://github.com/mrouvier/drbenchmark.git
cd DrBenchmark
pip install -r requirements.txt
```

## Jean-zay setup

```bash
module purge
module load pytorch-gpu/py3/1.12.1
```

```bash
git clone https://github.com/mrouvier/drbenchmark.git
cd DrBenchmark
pip install -r requirements.txt
```

## Add models to the benchmark

1. Open the `./models.txt` file.
2. Add HuggingFace remote or local path the models you are wanting to evaluate.

For example, we choose the following models:

```plain
Dr-BERT/DrBERT-7GB
camembert-base
almanach/camemberta-base
almanach/camembert-bio-base
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
flaubert/flaubert_base_uncased
```

## Run the benchmark

**On local SLURM:**

1. `sbatch run_all.sh`

**On Jean-Zay SLURM:**

1. `idr_compuse`
2. `nano run_all_jean_zay.sh` (replace the account identifier with the one from idr_compuse)
3. `sbatch run_all_jean_zay.sh`

## Tasks

|  **Dataset**           |  Train          | Validation            | Test                     | Task                                       | Metrics                       | HuggingFace                                                   |
|:----------------------:|:---------------:|:---------------------:|:------------------------:|:------------------------------------------:|:-----------------------------:|:-------------------------------------------------------------:|
| CAS - Task 1           |      5306       |        758            |         1516             | Part-Of-Speech Tagging (POS)               |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/CAS)           |
| CAS - Task 2           |      5306       |        758            |         1516             | Multi-class Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/CAS)           |
| ESSAI - Task 1         |      9693       |       1385            |         2770             | Part-Of-Speech Tagging (POS)               |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/ESSAI)         |
| ESSAI - Task 2         |      9693       |       1385            |         2770             | Multi-class Classification(CLS)            |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/ESSAI)         |
| QUAERO - EMEA          |      429        |       389             |         348              | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/QUAERO)        |
| QUAERO - MEDLINE       |      833        |       832             |         833              | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/QUAERO)        |
| E3C Task 1             |      969        |       140             |         293              | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/E3C)           |
| E3C Task 2             |      969        |       140             |         293              | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/E3C)           |
| MorFITT                |      1514       |       1022            |         1088             | Multi-label Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/MORFITT)       |
| FrenchMedMCQA - Task 1 |      2171       |        312            |         622              | Multiple-Choice Question Answering (MCQA)  |     Hamming Score / EMR       | [LINK](https://huggingface.co/datasets/Dr-BERT/FrenchMedMCQA) |
| FrenchMedMCQA - Task 2 |      2171       |        312            |         622              | Multi-class Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/FrenchMedMCQA) |
| Mantra-GSC - EMEA      |        70       |         10            |           20             | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/Mantra-GSC)    |
| Mantra-GSC - MEDLINE   |        70       |         10            |           20             | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/Mantra-GSC)    |
| Mantra-GSC - PATENT    |        35       |          5            |           10             | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/Mantra-GSC)    |
| CLISTER                |      499        |        101            |          400             | Semantic textual similarity (STS)          |     EDRM / Spearman           | [LINK](https://huggingface.co/datasets/Dr-BERT/CLISTER)       |
| DEFT-2019              |      3543       |        1340           |          7202            | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/DEFT2019)      |
| DEFT-2020 - Task 1     |      498        |        102            |          410             | Semantic textual similarity (STS)          |     EDRM / Spearman           | [LINK](https://huggingface.co/datasets/Dr-BERT/DEFT2020)      |
| DEFT-2020 - Task 2     |      460        |        112            |          530             | Multi-class Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/DEFT2020)      |
| DEFT-2021 - Task 1     |      118        |         49            |          108             | Multi-label Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/DEFT2021)      |
| DEFT-2021 - Task 1     |      2153       |         793           |          1766            | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/DEFT2021)      |
| DiaMED                 |        509      |         76            |           154            | Multi-class Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/DiaMed)        |
| PxCorpus               |        1386     |         198           |            397           | Named Entity Recognition (NER)             |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/PxCorpus)      |
| PxCorpus               |        1386     |         198           |            397           | Multi-class Classification (CLS)           |     F1                        | [LINK](https://huggingface.co/datasets/Dr-BERT/PxCorpus)      |

## Datasets

[CAS](http://natalia.grabar.free.fr/resources.php) is a corpus of 717 clinical cases annotated in both part-of-speech tagging and multi-class classification. 

[ESSAI](https://clementdalloux.fr/?page_id=28) is a corpus of 13,848 clinical cases annotated in both 41 part-of-speech tags and the same classes as CAS for the classification (NEGATION or SPECULATION). 

[QUAERO](https://quaerofrenchmed.limsi.fr/) The QUAERO French Medical Corpus introduces an extensive corpus of biomedical documents annotated at the entity and concept levels to provide NER and classification tasks. Three text genres are covered, comprising a total of 103,056 words obtained either from EMEA or MEDLINE. Ten entity categories corresponding to UMLS Semantic Groups were annotated, using automatic pre-annotations validated by trained human annotators. Overall, a total of 26,409 entity annotations were mapped to 5,797 unique UMLS concepts. To simplify the evaluation process, we sort the nested labels by alphabetical order and concatenate them together into a single one to transform the task to a usable format for token classification with BERT based architectures.

[E3C](https://github.com/hltfbk/E3C-Corpus) is a multilingual corpus of clinical narratives annotated for the named-entity recognition task.  It consists of two types of annotations: (i) clinical entities (e.g., pathologies), (ii) temporal information and factuality (e.g., events). It handles 5 languages (French, English, Italian, Spanish and Basque) but in our case, we are only interested by French.

[MorFITT](ddd) is the first multi-label corpus in French annotated in specialties in the medical field. It is composed of 3,624 abstracts of scientific articles from PubMed, annotated in 12 specialties for a total of 5,116 annotations.

[FrenchMedMCQA](https://aclanthology.org/2022.louhi-1.5.pdf) is a publicly available Multiple-Choice Question Answering (MCQA) dataset in French for medical domain. It contains 3,105 questions coming from real exams of the French medical specialization diploma in pharmacy, integrating single and multiple answers.

[Mantra-GSC](ddd) is a multilingual corpus annotated for biomedical named-entity recognition. It handles 5 languages (Spanish, French, German, Dutch and English) but in our case we are focusing only on the French subset. The corpus is obtained from three sources which as been partitioned to be evaluated separately since two annotation schemas are used: Medline (11 classes), EMEA and Patents (10 classes).

[CLISTER](https://aclanthology.org/2022.jeptalnrecital-taln.28.pdf) is a French clinical Semantic Textual Similarity (STS) corpus of 1,000 sentence pairs manually annotated by multiple annotators according to their similarity scores ranging from 0 to 5 and averaged together, giving a floating point number.

[DEFT-2019](https://aclanthology.org/2019.jeptalnrecital-deft.1/) is a corpus annotated for named-entity recognition proposed in the 2019 edition of the French Text Mining Challenge. This corpus is made of 717 clinical cases taken from the CAS corpus. The corpus is annotated with 4 entities: age, genre, issus and origine.

[DEFT-2020](https://aclanthology.org/2020.jeptalnrecital-deft.1.pdf) The 2020 edition of the French Text Mining Challenge proposed two tasks about textual similarity and one information extraction task. The first task aims at identifying the degree of similarity between pairs of sentences, from 0 (the less similar) to 5 (the most similar). The second task consists in identifying the closest sentence from a source sentence among three given sentences. Both tasks rely on a corpus from the general and health domains.

[DEFT-2021](https://aclanthology.org/2021.jeptalnrecital-deft.1/) is a corpus annotated on two tasks: (i) multi-label classification and (ii) named-entity recognition. This task aims at identifying patient's clinical profile on the basis of the diseases, signs or symptoms mentioned in a subset of 275 clinical cases taken from DEFT-2019. The corpus is annotated with 23 axes derived from Chapter C of Medical Subject Headings (MeSH)

[DiaMED]() is a corpus of XXX clinical cases manually annotated in major diagnostic categories (CMD). These CMD classes are a clinical classification composed of 28 classes, most often corresponding to a functional system (nervous system, eyes, ...) and allowing the assignment of patients to a particular hospital service.

[PxCorpus](https://zenodo.org/record/6524162) is the first open-sourced corpus of spoken language understanding on medical drug prescriptions. It contains 4 hours of transcribed and annotated dialogues of drug prescriptions in French acquired through an experiment with 55 experts and non-experts participants. In total, the dataset includes 1,981 recordings of 55 participants (38\% non-experts, 25\% doctors, 36\% medical practitioners), manually transcribed and semantically annotated. The first task consist of classifying the textual utterances into one of 4 intent classes (Appendix~\ref{sec:classes-pxcorpus}). The second one consist of a named entity recognition task where each word of a sequence is classified into one of the 38 classes and formatted at the IOB2 format.


## Citing DrBenchmark

```bibtex
TODO
```

## Citing DrBert

```bibtex
@misc{labrak2023drbert,
      title={DrBERT: A Robust Pre-trained Model in French for Biomedical and Clinical domains}, 
      author={Yanis Labrak and Adrien Bazoge and Richard Dufour and Mickael Rouvier and Emmanuel Morin and Béatrice Daille and Pierre-Antoine Gourraud},
      year={2023},
      eprint={2304.00958},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Citing datasets

CAS:

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

ESSAI:

```bibtex
@article{dalloux_claveau_grabar_oliveira_moro_gumiel_carvalho_2021,
  title="{Supervised learning for the detection of negation and of its scope in French and Brazilian Portuguese biomedical corpora}",
  volume={27},
  DOI={10.1017/S1351324920000352},
  number={2},
  journal={Natural Language Engineering},
  publisher={Cambridge University Press},
  author={Dalloux, Clément and Claveau, Vincent and Grabar, Natalia and Oliveira, Lucas Emanuel Silva and Moro, Claudia Maria Cabral and Gumiel, Yohan Bonescki and Carvalho, Deborah Ribeiro}, 
  year={2021},
  pages={181–201}
}
```

QUAERO:

```bibtex
@inproceedings{Nvol2014TheQF,
  title={The Quaero French Medical Corpus : A Ressource for Medical Entity Recognition and Normalization},
  author={Aur{\'e}lie N{\'e}v{\'e}ol and Cyril Grouin and J{\'e}r{\'e}my Leixa and Sophie Rosset and Pierre Zweigenbaum},
  year={2014}
}
```

E3C:

```bibtex
@article{Magnini2020TheEP,
  title={The E3C Project: Collection and Annotation of a Multilingual Corpus of Clinical Cases},
  author={Bernardo Magnini and Bego{\~n}a Altuna and Alberto Lavelli and Manuela Speranza and Roberto Zanoli},
  journal={Proceedings of the Seventh Italian Conference on Computational Linguistics CLiC-it 2020},
  year={2020}
}
```

MorFITT:
```bibtex
Comming Soon
```

FrenchMedMCQA:

```bibtex
@inproceedings{labrak:hal-03824241,
  TITLE = {{FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain}},
  AUTHOR = {Labrak, Yanis and Bazoge, Adrien and Dufour, Richard and Daille, B{\'e}atrice and Gourraud, Pierre-Antoine and Morin, Emmanuel and Rouvier, Mickael},
  URL = {https://hal.archives-ouvertes.fr/hal-03824241},
  BOOKTITLE = {{Proceedings of the 13th International Workshop on Health Text Mining and Information Analysis (LOUHI)}},
  ADDRESS = {Abou Dhabi, United Arab Emirates},
  YEAR = {2022},
  MONTH = Dec,
  PDF = {https://hal.archives-ouvertes.fr/hal-03824241/file/LOUHI_2022___QA-3.pdf},
  HAL_ID = {hal-03824241},
  HAL_VERSION = {v1},
}
```

Mantra-GSC:

```bibtex
@article{10.1093/jamia/ocv037,
    author = {Kors, Jan A and Clematide, Simon and Akhondi, Saber A and van Mulligen, Erik M and Rebholz-Schuhmann, Dietrich},
    title = "{A multilingual gold-standard corpus for biomedical concept recognition: the Mantra GSC}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {22},
    number = {5},
    pages = {948-956},
    year = {2015},
    month = {05},
    abstract = "{Objective To create a multilingual gold-standard corpus for biomedical concept recognition.Materials and methods We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups. To reduce the annotation workload, automatically generated preannotations were provided. Individual annotations were automatically harmonized and then adjudicated, and cross-language consistency checks were carried out to arrive at the final annotations.Results The number of final annotations was 5530. Inter-annotator agreement scores indicate good agreement (median F-score 0.79), and are similar to those between individual annotators and the gold standard. The automatically generated harmonized annotation set for each language performed equally well as the best annotator for that language.Discussion The use of automatic preannotations, harmonized annotations, and parallel corpora helped to keep the manual annotation efforts manageable. The inter-annotator agreement scores provide a reference standard for gauging the performance of automatic annotation techniques.Conclusion To our knowledge, this is the first gold-standard corpus for biomedical concept recognition in languages other than English. Other distinguishing features are the wide variety of semantic groups that are being covered, and the diversity of text genres that were annotated.}",
    issn = {1067-5027},
    doi = {10.1093/jamia/ocv037},
    url = {https://doi.org/10.1093/jamia/ocv037},
    eprint = {https://academic.oup.com/jamia/article-pdf/22/5/948/34146393/ocv037.pdf},
}
```

CLISTER:

```bibtex
@inproceedings{hiebel-etal-2022-clister-corpus,
    title = "{CLISTER} : A Corpus for Semantic Textual Similarity in {F}rench Clinical Narratives",
    author = {Hiebel, Nicolas  and
      Ferret, Olivier  and
      Fort, Kar{\"e}n  and
      N{\'e}v{\'e}ol, Aur{\'e}lie},
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.459",
    pages = "4306--4315",
    abstract = "Modern Natural Language Processing relies on the availability of annotated corpora for training and evaluating models. Such resources are scarce, especially for specialized domains in languages other than English. In particular, there are very few resources for semantic similarity in the clinical domain in French. This can be useful for many biomedical natural language processing applications, including text generation. We introduce a definition of similarity that is guided by clinical facts and apply it to the development of a new French corpus of 1,000 sentence pairs manually annotated according to similarity scores. This new sentence similarity corpus is made freely available to the community. We further evaluate the corpus through experiments of automatic similarity measurement. We show that a model of sentence embeddings can capture similarity with state-of-the-art performance on the DEFT STS shared task evaluation data set (Spearman=0.8343). We also show that the corpus is complementary to DEFT STS.",
}
```

DEFT-2019:

```bibtex
@inproceedings{Grabar-Grouin-Hamon-Claveau:DEFT:2019,
    author = "Grabar, Natalia and Grouin, Cyril and Hamon, Thierry and Claveau, Vincent",
    title = "Recherche et extraction d'information dans des cas cliniques. Pr\'esentation de la campagne d'\'evaluation DEFT 2019",
    booktitle = "Actes de la Conf\'erence sur le Traitement Automatique des Langues Naturelles (TALN)  PFIA 2019. D\'efi Fouille de Textes (atelier TALN-RECITAL)",
    month = "7",
    year = "2019",
    address = "Toulouse, France",
    publisher = "Association pour le Traitement Automatique des Langues",
    pages = "7-16",
    note = "Information Retrieval and Information Extraction from Clinical Cases",
    abstract = "Cet article pr\'esente la campagne d'\'evaluation DEFT 2019 sur l'analyse de textes cliniques r\'edig\'es en fran\c{c}ais. Le corpus se compose de cas cliniques publi\'es et discut\'es dans des articles scientifiques, et index\'es par des mots-cl\'es. Nous proposons trois t\^aches ind\'ependantes : l'indexation des cas cliniques et discussions, \'evalu\'ee prioritairement par la MAP (mean average precision), l'appariement entre cas cliniques et discussions, \'evalu\'e au moyen d'une pr\'ecision, et l'extraction d'information parmi quatre cat\'egories (\^age, genre, origine de la consultation, issue), \'evalu\'ee en termes de rappel, pr\'ecision et F-mesure. Nous pr\'esentons les r\'esultats obtenus par les participants sur chaque t\^ache.",
    keywords = "Cas clinique, fouille de texte, extraction d'information, recherche d'information,  \'evaluation.",
    url = "http://talnarchives.atala.org/ateliers/2019/DEFT/1.pdf"
}
```

DEFT-2020:

```bibtex
@inproceedings{cardon-etal-2020-presentation,
    title = "Pr{\'e}sentation de la campagne d{'}{\'e}valuation {DEFT} 2020 : similarit{\'e} textuelle en domaine ouvert et extraction d{'}information pr{\'e}cise dans des cas cliniques (Presentation of the {DEFT} 2020 Challenge : open domain textual similarity and precise information extraction from clinical cases )",
    author = "Cardon, R{\'e}mi  and
      Grabar, Natalia  and
      Grouin, Cyril  and
      Hamon, Thierry",
    booktitle = "Actes de la 6e conf{\'e}rence conjointe Journ{\'e}es d'{\'E}tudes sur la Parole (JEP, 33e {\'e}dition), Traitement Automatique des Langues Naturelles (TALN, 27e {\'e}dition), Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (R{\'E}CITAL, 22e {\'e}dition). Atelier D{\'E}fi Fouille de Textes",
    month = "6",
    year = "2020",
    address = "Nancy, France",
    publisher = "ATALA et AFCP",
    url = "https://aclanthology.org/2020.jeptalnrecital-deft.1",
    pages = "1--13",
    abstract = "L{'}{\'e}dition 2020 du d{\'e}fi fouille de texte (DEFT) a propos{\'e} deux t{\^a}ches autour de la similarit{\'e} textuelle et une t{\^a}che d{'}extraction d{'}information. La premi{\`e}re t{\^a}che vise {\`a} identifier le degr{\'e} de similarit{\'e} entre paires de phrases sur une {\'e}chelle de 0 (le moins similaire) {\`a} 5 (le plus similaire). Les r{\'e}sultats varient de 0,65 {\`a} 0,82 d{'}EDRM. La deuxi{\`e}me t{\^a}che consiste {\`a} d{\'e}terminer la phrase la plus proche d{'}une phrase source parmi trois phrases cibles fournies, avec des r{\'e}sultats tr{\`e}s {\'e}lev{\'e}s, variant de 0,94 {\`a} 0,99 de pr{\'e}cision. Ces deux t{\^a}ches reposent sur un corpus du domaine g{\'e}n{\'e}ral et de sant{\'e}. La troisi{\`e}me t{\^a}che propose d{'}extraire dix cat{\'e}gories d{'}informations du domaine m{\'e}dical depuis le corpus de cas cliniques de DEFT 2019. Les r{\'e}sultats varient de 0,07 {\`a} 0,66 de F-mesure globale pour la sous-t{\^a}che des pathologies et signes ou sympt{\^o}mes, et de 0,14 {\`a} 0,76 pour la sous-t{\^a}che sur huit cat{\'e}gories m{\'e}dicales. Les m{\'e}thodes utilis{\'e}es reposent sur des CRF et des r{\'e}seaux de neurones.",
    language = "French",
}
```

DEFT-2021:

```bibtex
@inproceedings{grouin-etal-2021-classification,
    title = "Classification de cas cliniques et {\'e}valuation automatique de r{\'e}ponses d{'}{\'e}tudiants : pr{\'e}sentation de la campagne {DEFT} 2021 (Clinical cases classification and automatic evaluation of student answers : Presentation of the {DEFT} 2021 Challenge)",
    author = "Grouin, Cyril  and
      Grabar, Natalia  and
      Illouz, Gabriel",
    booktitle = "Actes de la 28e Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles. Atelier D{\'E}fi Fouille de Textes (DEFT)",
    month = "6",
    year = "2021",
    address = "Lille, France",
    publisher = "ATALA",
    url = "https://aclanthology.org/2021.jeptalnrecital-deft.1",
    pages = "1--13",
    abstract = "Le d{\'e}fi fouille de textes (DEFT) est une campagne d{'}{\'e}valuation annuelle francophone. Nous pr{\'e}sentons les corpus et baselines {\'e}labor{\'e}es pour trois t{\^a}ches : (i) identifier le profil clinique de patients d{\'e}crits dans des cas cliniques, (ii) {\'e}valuer automatiquement les r{\'e}ponses d{'}{\'e}tudiants sur des questionnaires en ligne (Moodle) {\`a} partir de la correction de l{'}enseignant, et (iii) poursuivre une {\'e}valuation de r{\'e}ponses d{'}{\'e}tudiants {\`a} partir de r{\'e}ponses d{\'e}j{\`a} {\'e}valu{\'e}es par l{'}enseignant. Les r{\'e}sultats varient de 0,394 {\`a} 0,814 de F-mesure sur la premi{\`e}re t{\^a}che (7 {\'e}quipes), de 0,448 {\`a} 0,682 de pr{\'e}cision sur la deuxi{\`e}me (3 {\'e}quipes), et de 0,133 {\`a} 0,510 de pr{\'e}cision sur la derni{\`e}re (3 {\'e}quipes).",
    language = "French",
}
```

DiaMED:

```bibtex
TODO
```

PxCorpus:

```bibtex
@InProceedings{Kocabiyikoglu2022,
  author =     "Alican Kocabiyikoglu and Fran{\c c}ois Portet and Prudence Gibert and Hervé Blanchon and Jean-Marc Babouchkine and Gaëtan Gavazzi",
  title =     "A Spoken Drug Prescription Dataset in French for Spoken Language Understanding",
  booktitle =     "13th Language Resources and Evaluation Conference (LREC 2022)",
  year =     "2022",
  location =     "Marseille, France"
}
```

## Acknowledgments

This work was performed using HPC resources from GENCI-IDRIS (Grant 2022-AD011013061R1 and 2022-AD011013715) and from CCIPL (Centre de Calcul Intensif des Pays de la Loire). This work was financially supported by ANR AIBy4 (ANR-20-THIA-0011) and Zenidoc.
