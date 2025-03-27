import os
import json
import random

import datasets
import numpy as np
import pandas as pd

_CITATION = """\
ddd
"""

_DESCRIPTION = """\
ddd
"""

_HOMEPAGE = "ddd"

_URL = "https://huggingface.co/datasets/Dr-BERT/MORFITT/resolve/main/data.zip"

_LICENSE = "unknown"

_SPECIALITIES = ['microbiology', 'etiology', 'virology', 'physiology', 'immunology', 'parasitology', 'genetics', 'chemistry', 'veterinary', 'surgery', 'pharmacology', 'psychology']


class MORFITT(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "source"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="source", version="1.0.0", description="The MORFITT corpora"),
    ]

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "abstract": datasets.Value("string"),
                "specialities": datasets.Sequence(
                    datasets.features.ClassLabel(names=_SPECIALITIES),
                ),
                "specialities_one_hot": datasets.Sequence(
                    datasets.Value("float"),
                ),
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "tsv_file": data_dir + "/train.tsv",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "tsv_file": data_dir + "/dev.tsv",
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "tsv_file": data_dir + "/test.tsv",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, tsv_file, split):

        # Load TSV file
        df = pd.read_csv(tsv_file, sep="\t")

        for index, e in df.iterrows():

            specialities = e["specialities"].split("|")

            # Empty one hot vector
            one_hot = [0.0 for i in _SPECIALITIES]

            # Fill up the one hot vector
            for s in specialities:
                one_hot[_SPECIALITIES.index(s)] = 1.0

            yield e["identifier"], {
                "id": e["identifier"],
                "abstract": e["abstract"],
                "specialities": specialities,
                "specialities_one_hot": one_hot,
            }
