import os
import random

import datasets
import numpy as np

_CITATION = """\
 @misc{
    dalloux,
    title={Datasets – Clément Dalloux},
    url={http://clementdalloux.fr/?page_id=28},
    journal={Clément Dalloux},
    author={Dalloux, Clément}
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

This version only contain the annotated ESSAI corpus
"""

_HOMEPAGE = "https://clementdalloux.fr/?page_id=28"

_LICENSE = 'Data User Agreement'

_URL = "https://drbenchmark.univ-avignon.fr/corpus/cas_essai.zip"


class ESSAI(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "pos"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="pos", version="1.0.0", description="The ESSAI corpora - POS Speculation task"),

        datasets.BuilderConfig(name="cls", version="1.0.0", description="The ESSAI corpora - CLS Negation / Speculation task"),

        datasets.BuilderConfig(name="ner_spec", version="1.0.0", description="The ESSAI corpora - NER Speculation task"),
        datasets.BuilderConfig(name="ner_neg", version="1.0.0", description="The ESSAI corpora - NER Negation task"),
    ]

    def _info(self):

        if self.config.name.find("pos") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "lemmas": [datasets.Value("string")],
                    "pos_tags": [datasets.features.ClassLabel(
                        names=['B-INT', 'B-PRO:POS', 'B-PRP', 'B-SENT', 'B-PRO', 'B-ABR', 'B-VER:pres', 'B-KON', 'B-SYM', 'B-DET:POS', 'B-VER:', 'B-PRO:IND', 'B-NAM', 'B-ADV', 'B-PRO:DEM', 'B-NN', 'B-PRO:PER', 'B-VER:pper', 'B-VER:ppre', 'B-PUN', 'B-VER:simp', 'B-PREF', 'B-NUM', 'B-VER:futu', 'B-NOM', 'B-VER:impf', 'B-VER:subp', 'B-VER:infi', 'B-DET:ART', 'B-PUN:cit', 'B-ADJ', 'B-PRP:det', 'B-PRO:REL', 'B-VER:cond', 'B-VER:subi'],
                    )],
                }
            )

        elif self.config.name.find("cls") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "label": datasets.features.ClassLabel(
                        names=['negation_speculation', 'negation', 'neutral', 'speculation'],
                    ),
                }
            )

        elif self.config.name.find("ner") != -1:

            if self.config.name.find("_spec") != -1:
                names = ['O', 'B_cue_spec', 'B_scope_spec', 'I_scope_spec']
            elif self.config.name.find("_neg") != -1:
                names = ['O', 'B_cue_neg', 'B_scope_neg', 'I_scope_neg']

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": [datasets.Value("string")],
                    "lemmas": [datasets.Value("string")],
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
            docs = [f"corpora/ESSAI_{subset}.txt"]
        else:
            docs = ["corpora/ESSAI_neg.txt", "corpora/ESSAI_spec.txt"]

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
                        elif tag == "v":
                            tag = "I_scope_spec"
                        elif tag == "z":
                            tag = "O"
                        elif tag == "I_scope_spec_":
                            tag = "I_scope_spec"

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
