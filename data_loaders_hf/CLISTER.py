import os
import json
import random

import datasets
import numpy as np
import pandas as pd

_CITATION = """\
@inproceedings{hiebel:cea-03740484,
  TITLE = {{CLISTER: A corpus for semantic textual similarity in French clinical narratives}},
  AUTHOR = {Hiebel, Nicolas and Ferret, Olivier and Fort, Kar{\"e}n and N{\'e}v{\'e}ol, Aur{\'e}lie},
  URL = {https://hal-cea.archives-ouvertes.fr/cea-03740484},
  BOOKTITLE = {{LREC 2022 - 13th Language Resources and Evaluation Conference}},
  ADDRESS = {Marseille, France},
  PUBLISHER = {{European Language Resources Association}},
  SERIES = {LREC 2022 - Proceedings of the 13th Conference on Language Resources and Evaluation},
  VOLUME = {2022},
  PAGES = {4306‑4315},
  YEAR = {2022},
  MONTH = Jun,
  KEYWORDS = {Semantic Similarity ; Corpus Development ; Clinical Text ; French ; Semantic Similarity},
  PDF = {https://hal-cea.archives-ouvertes.fr/cea-03740484/file/2022.lrec-1.459.pdf},
  HAL_ID = {cea-03740484},
  HAL_VERSION = {v1},
}
"""

_DESCRIPTION = """\
Modern Natural Language Processing relies on the availability of annotated corpora for training and \
evaluating models. Such resources are scarce, especially for specialized domains in languages other \
than English. In particular, there are very few resources for semantic similarity in the clinical domain \
in French. This can be useful for many biomedical natural language processing applications, including \
text generation. We introduce a definition of similarity that is guided by clinical facts and apply it \
to the development of a new French corpus of 1,000 sentence pairs manually annotated according to \
similarity scores. This new sentence similarity corpus is made freely available to the community. We \
further evaluate the corpus through experiments of automatic similarity measurement. We show that a \
model of sentence embeddings can capture similarity with state of the art performance on the DEFT STS \
shared task evaluation data set (Spearman=0.8343). We also show that the CLISTER corpus is complementary \
to DEFT STS. \
"""

_HOMEPAGE = "https://gitlab.inria.fr/codeine/clister"

_LICENSE = "unknown"

_URL = "https://drbenchmark.univ-avignon.fr/corpus/clister.tar.gz"


class CLISTER(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "source"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="source", version="1.0.0", description="The CLISTER corpora"),
    ]

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "document_1_id": datasets.Value("string"),
                "document_2_id": datasets.Value("string"),
                "text_1": datasets.Value("string"),
                "text_2": datasets.Value("string"),
                "label": datasets.Value("float"),
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

        # data_dir = self.config.data_dir.rstrip("/")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "csv_file": data_dir + "/train.csv",
                    "json_file": data_dir + "/id_to_sentence_train.json",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "csv_file": data_dir + "/train.csv",
                    "json_file": data_dir + "/id_to_sentence_train.json",
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "csv_file": data_dir + "/test.csv",
                    "json_file": data_dir + "/id_to_sentence_test.json",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, csv_file, json_file, split):

        all_res = []

        key = 0

        # Load JSON file
        with open(json_file) as f_json:
            data_map = json.load(f_json)

        # Load CSV file
        df = pd.read_csv(csv_file, sep="\t")

        for index, e in df.iterrows():

            all_res.append({
                "id": str(key),
                "document_1_id": e["id_1"],
                "document_2_id": e["id_2"],
                "text_1": data_map["_".join(e["id_1"].split("_")[0:2])],
                "text_2": data_map["_".join(e["id_2"].split("_")[0:2])],
                "label": float(e["sim"]),
            })

            key += 1

        if split != "test":

            ids = [r["id"] for r in all_res]

            random.seed(4)
            random.shuffle(ids)
            random.shuffle(ids)
            random.shuffle(ids)

            train, validation = np.split(ids, [int(len(ids) * 0.8333)])

            if split == "train":
                allowed_ids = list(train)
            elif split == "validation":
                allowed_ids = list(validation)

            for r in all_res:
                if r["id"] in allowed_ids:
                    yield r["id"], r
        else:

            for r in all_res:
                yield r["id"], r
