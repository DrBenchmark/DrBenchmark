# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DIAMED"""

import os
import json
import math

import datasets

_DESCRIPTION = """\
DIAMED
"""

_HOMEPAGE = "DIAMED"

_LICENSE = "Apache License 2.0"

_URL = "https://huggingface.co/datasets/Dr-BERT/DiaMED/resolve/main/data.zip"

_CITATION = """\
DIAMED
"""


class DiaMed(datasets.GeneratorBasedBuilder):
    """DIAMED"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"default", version="1.0.0", description=f"DiaMED data"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):

        features = datasets.Features(
            {
                "identifier": datasets.Value("string"),
                "title": datasets.Value("string"),
                "clinical_case": datasets.Value("string"),
                "topic": datasets.Value("string"),
                "keywords": datasets.Sequence(
                    datasets.Value("string"),
                ),
                "domains": datasets.Sequence(
                    datasets.Value("string"),
                ),
                "collected_at": datasets.Value("string"),
                "published_at": datasets.Value("string"),
                "source_url": datasets.Value("string"),
                "source_name": datasets.Value("string"),
                "license": datasets.Value("string"),
                "figures_urls": datasets.Sequence(
                    datasets.Value("string"),
                ),
                "figures_paths": datasets.Sequence(
                    datasets.Value("string"),
                ),
                "figures": datasets.Sequence(
                    datasets.Image(),
                ),
                "icd-10": datasets.features.ClassLabel(names=[
                    'A00-B99  Certain infectious and parasitic diseases',
                    'C00-D49  Neoplasms',
                    'D50-D89  Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
                    'E00-E89  Endocrine, nutritional and metabolic diseases',
                    'F01-F99  Mental, Behavioral and Neurodevelopmental disorders',
                    'G00-G99  Diseases of the nervous system',
                    'H00-H59  Diseases of the eye and adnexa',
                    'H60-H95  Diseases of the ear and mastoid process',
                    'I00-I99  Diseases of the circulatory system',
                    'J00-J99  Diseases of the respiratory system',
                    'K00-K95  Diseases of the digestive system',
                    'L00-L99  Diseases of the skin and subcutaneous tissue',
                    'M00-M99  Diseases of the musculoskeletal system and connective tissue',
                    'N00-N99  Diseases of the genitourinary system',
                    'O00-O9A  Pregnancy, childbirth and the puerperium',
                    'P00-P96  Certain conditions originating in the perinatal period',
                    'Q00-Q99  Congenital malformations, deformations and chromosomal abnormalities',
                    'R00-R99  Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
                    'S00-T88  Injury, poisoning and certain other consequences of external causes',
                    'U00-U85  Codes for special purposes',
                    'V00-Y99  External causes of morbidity',
                    'Z00-Z99  Factors influencing health status and contact with health services',
                ]),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "base_path": data_dir,
                    "filepath": data_dir + "/splits/train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "base_path": data_dir,
                    "filepath": data_dir + "/splits/validation.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "base_path": data_dir,
                    "filepath": data_dir + "/splits/test.json",
                },
            ),
        ]

    def _generate_examples(self, base_path, filepath):

        with open(filepath, encoding="utf-8") as f:

            data = json.load(f)

            for key, d in enumerate(data):

                if str(d["icd-10"]) == "nan" or d["icd-10"].find("Plusieurs cas cliniques") != -1 or d["icd-10"].find("Aucune annotation") != -1:
                    continue

                yield key, {
                    "identifier": d["identifier"],
                    "title": d["title"],
                    "clinical_case": d["clinical_case"],
                    "topic": d["topic"],
                    "keywords": d["keywords"],
                    "domains": d["domain"],
                    "collected_at": d["collected_at"],
                    "published_at": d["published_at"],
                    "source_url": d["source_url"],
                    "source_name": d["source_name"],
                    "license": d["license"],
                    "figures_urls": d["figures"],
                    "figures": [base_path + fg.lstrip(".") for fg in d["local_figures"]],
                    "figures_paths": [base_path + fg.lstrip(".") for fg in d["local_figures"]],
                    "icd-10": d["icd-10"],
                }
