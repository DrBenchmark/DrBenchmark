import os
import random

import datasets
import numpy as np

_CITATION = """\
@inproceedings{grabar-etal-2018-cas,
  title        = {{CAS}: {F}rench Corpus with Clinical Cases},
  author       = {Grabar, Natalia  and Claveau, Vincent  and Dalloux, Cl{\'e}ment},
  year         = 2018,
  month        = oct,
  booktitle    = {
    Proceedings of the Ninth International Workshop on Health Text Mining and
    Information Analysis
  },
  publisher    = {Association for Computational Linguistics},
  address      = {Brussels, Belgium},
  pages        = {122--128},
  doi          = {10.18653/v1/W18-5614},
  url          = {https://aclanthology.org/W18-5614},
  abstract     = {
    Textual corpora are extremely important for various NLP applications as
    they provide information necessary for creating, setting and testing these
    applications and the corresponding tools. They are also crucial for
    designing reliable methods and reproducible results. Yet, in some areas,
    such as the medical area, due to confidentiality or to ethical reasons, it
    is complicated and even impossible to access textual data representative of
    those produced in these areas. We propose the CAS corpus built with
    clinical cases, such as they are reported in the published scientific
    literature in French. We describe this corpus, currently containing over
    397,000 word occurrences, and the existing linguistic and semantic
    annotations.
  }
}
"""

_DESCRIPTION = """\
We manually annotated two corpora from the biomedical field. The ESSAI corpus \
contains clinical trial protocols in French. They were mainly obtained from the \
National Cancer Institute The typical protocol consists of two parts: the \
summary of the trial, which indicates the purpose of the trial and the methods \
applied; and a detailed description of the trial with the inclusion and \
exclusion criteria. The CAS corpus contains clinical cases published in \
scientific literature and training material. They are published in different \
journals from French-speaking countries (France, Belgium, Switzerland, Canada, \
African countries, tropical countries) and are related to various medical \
specialties (cardiology, urology, oncology, obstetrics, pulmonology, \
gastro-enterology). The purpose of clinical cases is to describe clinical \
situations of patients. Hence, their content is close to the content of clinical \
narratives (description of diagnoses, treatments or procedures, evolution, \
family history, expected audience, etc.). In clinical cases, the negation is \
frequently used for describing the patient signs, symptoms, and diagnosis. \
Speculation is present as well but less frequently.
This version only contain the annotated CAS corpus
"""

_HOMEPAGE = "https://clementdalloux.fr/?page_id=28"

_URL = "https://drbenchmark.univ-avignon.fr/corpus/cas_essai.zip"

_LICENSE = 'Data User Agreement'


class CAS(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "pos_spec"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="pos", version="1.0.0", description="The CAS corpora - POS Speculation task"),

        datasets.BuilderConfig(name="cls", version="1.0.0", description="The CAS corpora - CLS Negation / Speculation task"),

        datasets.BuilderConfig(name="ner_spec", version="1.0.0", description="The CAS corpora - NER Speculation task"),
        datasets.BuilderConfig(name="ner_neg", version="1.0.0", description="The CAS corpora - NER Negation task"),
    ]

    def _info(self):

        if self.config.name.find("pos") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "lemmas": [datasets.Value("string")],
                    # "pos_tags": [datasets.Value("string")],
                    "pos_tags": [datasets.features.ClassLabel(
                        names=['B-INT', 'B-PRO:DEM', 'B-VER:impf', 'B-VER:ppre', 'B-PRP:det', 'B-KON', 'B-VER:pper', 'B-PRP', 'B-PRO:IND', 'B-VER:simp', 'B-VER:con', 'B-SENT', 'B-VER:futu', 'B-PRO:PER', 'B-VER:infi', 'B-ADJ', 'B-NAM', 'B-NUM', 'B-PUN:cit', 'B-PRO:REL', 'B-VER:subi', 'B-ABR', 'B-NOM', 'B-VER:pres', 'B-DET:ART', 'B-VER:cond', 'B-VER:subp', 'B-DET:POS', 'B-ADV', 'B-SYM', 'B-PUN'],
                    )],
                }
            )

        elif self.config.name.find("cls") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    # "label": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=['negation_speculation', 'speculation', 'neutral', 'negation'],
                    ),
                }
            )

        elif self.config.name.find("ner") != -1:

            if self.config.name.find("_spec") != -1:
                names = ['O', 'B_xcope_inc', 'I_xcope_inc']
            elif self.config.name.find("_neg") != -1:
                names = ['O', 'B_scope_neg', 'I_scope_neg']

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "lemmas": [datasets.Value("string")],
                    # "ner_tags": [datasets.Value("string")],
                    "ner_tags": [datasets.features.ClassLabel(
                        names=names,
                    )],
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL).rstrip("/")

        '''
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")

        else:
            data_dir = self.config.data_dir
        '''
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "datadir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "datadir": data_dir,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "datadir": data_dir,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, datadir, split):

        all_res = []

        key = 0

        subset = self.config.name.split("_")[-1]

        unique_id_doc = []

        if self.config.name.find("ner") != -1:
            docs = [f"corpora/CAS_{subset}.txt"]
        else:
            docs = ["corpora/CAS_neg.txt", "corpora/CAS_spec.txt"]

        for file in docs:

            filename = os.path.join(datadir, file)

            if self.config.name.find("pos") != -1:

                id_docs = []
                id_words = []
                words = []
                lemmas = []
                POS_tags = []

                with open(filename) as f:

                    for line in f.readlines():

                        splitted = line.split("\t")

                        if len(splitted) < 5:
                            continue

                        id_doc, id_word, word, lemma, tag = splitted[0:5]
                        if len(splitted) >= 8:
                            tag = splitted[6]

                        if tag == "@card@":
                            print(splitted)

                        if word == "@card@":
                            print(splitted)

                        if lemma == "000" and tag == "@card@":
                            tag = "NUM"
                            word = "100 000"
                            lemma = "100 000"
                        elif lemma == "45" and tag == "@card@":
                            tag = "NUM"

                        # if id_doc in id_docs:
                        #     continue

                        id_docs.append(id_doc)
                        id_words.append(id_word)
                        words.append(word)
                        lemmas.append(lemma)
                        POS_tags.append('B-' + tag)

                dic = {
                    "id_docs": np.array(list(map(int, id_docs))),
                    "id_words": id_words,
                    "words": words,
                    "lemmas": lemmas,
                    "POS_tags": POS_tags,
                }

                for doc_id in set(dic["id_docs"]):

                    indexes = np.argwhere(dic["id_docs"] == doc_id)[:, 0]
                    tokens = [dic["words"][id] for id in indexes]
                    text_lemmas = [dic["lemmas"][id] for id in indexes]
                    pos_tags = [dic["POS_tags"][id] for id in indexes]

                    if doc_id not in unique_id_doc:

                        all_res.append({
                            "id": str(doc_id),
                            "document_id": doc_id,
                            "tokens": tokens,
                            "lemmas": text_lemmas,
                            "pos_tags": pos_tags,
                        })
                        unique_id_doc.append(doc_id)

                        # key += 1

            elif self.config.name.find("ner") != -1:

                id_docs = []
                id_words = []
                words = []
                lemmas = []
                ner_tags = []

                with open(filename) as f:

                    for line in f.readlines():

                        if len(line.split("\t")) < 5:
                            continue

                        id_doc, id_word, word, lemma, _ = line.split("\t")[0:5]
                        tag = line.replace("\n", "").split("\t")[-1]

                        if tag == "***" or tag == "_":
                            tag = "O"
                        elif tag == "I_xcope_inc_":
                            tag = "I_xcope_inc"
                        # elif tag == "v":
                        #     tag = "I_scope_spec"
                        # elif tag == "z":
                        #     tag = "O"

                        id_docs.append(id_doc)
                        id_words.append(id_word)
                        words.append(word)
                        lemmas.append(lemma)
                        ner_tags.append(tag)

                dic = {
                    "id_docs": np.array(list(map(int, id_docs))),
                    "id_words": id_words,
                    "words": words,
                    "lemmas": lemmas,
                    "ner_tags": ner_tags,
                }

                for doc_id in set(dic["id_docs"]):

                    indexes = np.argwhere(dic["id_docs"] == doc_id)[:, 0]
                    tokens = [dic["words"][id] for id in indexes]
                    text_lemmas = [dic["lemmas"][id] for id in indexes]
                    ner_tags = [dic["ner_tags"][id] for id in indexes]

                    all_res.append({
                        "id": key,
                        "document_id": doc_id,
                        "tokens": tokens,
                        "lemmas": text_lemmas,
                        "ner_tags": ner_tags,
                    })

                    key += 1

            elif self.config.name.find("cls") != -1:

                with open(filename) as f_in:
                    conll = [
                        [b.split("\t") for b in a.split("\n")]
                        for a in f_in.read().split("\n\n")
                    ]

                classe = "negation" if filename.find("_neg") != -1 else "speculation"

                for document in conll:

                    if document == [""]:
                        continue

                    identifier = document[0][0]

                    unique = list(set([w[-1] for w in document]))
                    tokens = [sent[2] for sent in document if len(sent) > 1]

                    if "***" in unique:
                        l = "neutral"
                    elif "_" in unique:
                        l = classe

                    if identifier in unique_id_doc and l == 'neutral':
                        continue

                    elif identifier in unique_id_doc and l != 'neutral':

                        index_l = unique_id_doc.index(identifier)

                        if all_res[index_l]["label"] != "neutral":
                            l = "negation_speculation"

                        all_res[index_l] = {
                            "id": str(identifier),
                            "document_id": identifier,
                            "tokens": tokens,
                            "label": l,
                        }

                    else:

                        all_res.append({
                            "id": str(identifier),
                            "document_id": identifier,
                            "tokens": tokens,
                            "label": l,
                        })

                        unique_id_doc.append(identifier)

        ids = [r["id"] for r in all_res]

        random.seed(4)
        random.shuffle(ids)
        random.shuffle(ids)
        random.shuffle(ids)

        train, validation, test = np.split(ids, [int(len(ids) * 0.70), int(len(ids) * 0.80)])

        if split == "train":
            allowed_ids = list(train)
        elif split == "validation":
            allowed_ids = list(validation)
        elif split == "test":
            allowed_ids = list(test)

        for r in all_res:
            if r["id"] in allowed_ids:
                yield r["id"], r
