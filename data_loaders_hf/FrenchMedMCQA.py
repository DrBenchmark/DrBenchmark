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
"""FrenchMedMCQA : A French Multiple-Choice Question Answering Corpus for Medical domain"""

import os
import json

import datasets

_DESCRIPTION = """\
This paper introduces FrenchMedMCQA, the first publicly available Multiple-Choice \
Question Answering (MCQA) dataset in French for medical domain. It is composed of  \
3,105 questions taken from real exams of the French medical specialization diploma \
in pharmacy, mixing single and multiple answers. Each instance of the dataset contains \
an identifier, a question, five possible answers and their manual correction(s).  \
We also propose first baseline models to automatically process this MCQA task in  \
order to report on the current performances and to highlight the difficulty of the  \
task. A detailed analysis of the results showed that it is necessary to have  \
representations adapted to the medical domain or to the MCQA task: in our case,  \
English specialized models yielded better results than generic French ones, even though  \
FrenchMedMCQA is in French. Corpus, models and tools are available online.
"""

_HOMEPAGE = "https://frenchmedmcqa.github.io"

_LICENSE = "Apache License 2.0"

_URL = "https://huggingface.co/datasets/Dr-BERT/FrenchMedMCQA/resolve/main/data.zip"

_CITATION = """\
@unpublished{labrak:hal-03824241,
  TITLE = {{FrenchMedMCQA: A French Multiple-Choice Question Answering Dataset for Medical domain}},
  AUTHOR = {Labrak, Yanis and Bazoge, Adrien and Dufour, Richard and Daille, Béatrice and Gourraud, Pierre-Antoine and Morin, Emmanuel and Rouvier, Mickael},
  URL = {https://hal.archives-ouvertes.fr/hal-03824241},
  NOTE = {working paper or preprint},
  YEAR = {2022},
  MONTH = Oct,
  PDF = {https://hal.archives-ouvertes.fr/hal-03824241/file/LOUHI_2022___QA-3.pdf},
  HAL_ID = {hal-03824241},
  HAL_VERSION = {v1},
}
"""

class FrenchMedMCQA(datasets.GeneratorBasedBuilder):
    """FrenchMedMCQA : A French Multi-Choice Question Answering Corpus for Medical domain"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer_a": datasets.Value("string"),
                "answer_b": datasets.Value("string"),
                "answer_c": datasets.Value("string"),
                "answer_d": datasets.Value("string"),
                "answer_e": datasets.Value("string"),
                "correct_answers": datasets.Sequence(
                    datasets.features.ClassLabel(names=["a", "b", "c", "d", "e"]),
                ),
                "type": datasets.Value("string"),
                "subject_name": datasets.Value("string"),
                "number_correct_answers": datasets.features.ClassLabel(names=["1","2","3","4","5"]),
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

        data_dir = dl_manager.download_and_extract(_URL).rstrip("/")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir + "/train.json",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir + "/dev.json",
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir + "/test.json",
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):

        with open(filepath, encoding="utf-8") as f:

            data = json.load(f)

            for key, d in enumerate(data):

                yield key, {
                    "id": d["id"],
                    "question": d["question"],
                    "answer_a": d["answers"]["a"],
                    "answer_b": d["answers"]["b"],
                    "answer_c": d["answers"]["c"],
                    "answer_d": d["answers"]["d"],
                    "answer_e": d["answers"]["e"],
                    "correct_answers": d["correct_answers"],
                    "number_correct_answers": str(len(d["correct_answers"])),
                    "type": d["type"] if split != "test" else "unknown",
                    "subject_name": d["subject_name"],
                }
