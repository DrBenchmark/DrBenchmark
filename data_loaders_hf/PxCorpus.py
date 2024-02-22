# pip install bs4 syntok

import os
import random

import datasets

import numpy as np
from bs4 import BeautifulSoup, ResultSet
from syntok.tokenizer import Tokenizer

tokenizer = Tokenizer()

_CITATION = """\
@InProceedings{Kocabiyikoglu2022,
  author =     "Alican Kocabiyikoglu and Fran{\c c}ois Portet and Prudence Gibert and Hervé Blanchon and Jean-Marc Babouchkine and Gaëtan Gavazzi",
  title =     "A Spoken Drug Prescription Dataset in French for Spoken Language Understanding",
  booktitle =     "13th Language Resources and Evaluation Conference (LREC 2022)",
  year =     "2022",
  location =     "Marseille, France"
}
"""

_DESCRIPTION = """\
PxSLU is to the best of our knowledge, the first spoken medical drug prescriptions corpus to be distributed. It contains 4 hours of transcribed
and annotated dialogues of drug prescriptions in French acquired through an experiment with 55 participants experts and non-experts in drug prescriptions.

The automatic transcriptions were verified by human effort and aligned with semantic labels to allow training of NLP models. The data acquisition
protocol was reviewed by medical experts and permit free distribution without breach of privacy and regulation.

Overview of the Corpus

The experiment has been performed in wild conditions with naive participants and medical experts. In total, the dataset includes 1981 recordings
of 55 participants (38% non-experts, 25% doctors, 36% medical practitioners), manually transcribed and semantically annotated.
"""

_URL = "https://zenodo.org/record/6524162/files/pxslu.zip?download=1"

class PxCorpus(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"default", version="1.0.0", description=f"PxCorpus data"),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(
                    names=["medical_prescription", "negate", "none", "replace"],
                ),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=['O', 'B-A', 'B-cma_event', 'B-d_dos_form', 'B-d_dos_form_ext', 'B-d_dos_up', 'B-d_dos_val', 'B-dos_cond', 'B-dos_uf', 'B-dos_val', 'B-drug', 'B-dur_ut', 'B-dur_val', 'B-fasting', 'B-freq_days', 'B-freq_int_v1', 'B-freq_int_v1_ut', 'B-freq_int_v2', 'B-freq_int_v2_ut', 'B-freq_startday', 'B-freq_ut', 'B-freq_val', 'B-inn', 'B-max_unit_uf', 'B-max_unit_ut', 'B-max_unit_val', 'B-min_gap_ut', 'B-min_gap_val', 'B-qsp_ut', 'B-qsp_val', 'B-re_ut', 'B-re_val', 'B-rhythm_hour', 'B-rhythm_perday', 'B-rhythm_rec_ut', 'B-rhythm_rec_val', 'B-rhythm_tdte', 'B-roa', 'I-cma_event', 'I-d_dos_form', 'I-d_dos_form_ext', 'I-d_dos_up', 'I-d_dos_val', 'I-dos_cond', 'I-dos_uf', 'I-dos_val', 'I-drug', 'I-fasting', 'I-freq_startday', 'I-inn', 'I-rhythm_tdte', 'I-roa'],
                    ),
                ),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):

        data_dir = dl_manager.download_and_extract(_URL)

        print(data_dir)
            
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath_1": os.path.join(data_dir, "seq.in"),
                    "filepath_2": os.path.join(data_dir, "seq.label"),
                    "filepath_3": os.path.join(data_dir, "PxSLU_conll.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath_1": os.path.join(data_dir, "seq.in"),
                    "filepath_2": os.path.join(data_dir, "seq.label"),
                    "filepath_3": os.path.join(data_dir, "PxSLU_conll.txt"),
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath_1": os.path.join(data_dir, "seq.in"),
                    "filepath_2": os.path.join(data_dir, "seq.label"),
                    "filepath_3": os.path.join(data_dir, "PxSLU_conll.txt"),
                    "split": "test",
                },
            ),
        ]

    def getTokenTags(self, document):

        tokens = []
        ner_tags = []

        for pair in document.split("\n"):

            if len(pair) <= 0:
                continue

            text, label = pair.split("\t")
            tokens.append(text)
            ner_tags.append(label)

        return tokens, ner_tags

    def _generate_examples(self, filepath_1, filepath_2, filepath_3, split):

        key = 0
        all_res = []
    
        f_seq_in = open(filepath_1, "r")
        seq_in = f_seq_in.read().split("\n")
        f_seq_in.close()

        f_seq_label = open(filepath_2, "r")
        seq_label = f_seq_label.read().split("\n")
        f_seq_label.close()

        f_in_ner = open(filepath_3, "r")
        docs = f_in_ner.read().split("\n\n")
        f_in_ner.close()

        for idx, doc in enumerate(docs):

            text = seq_in[idx]
            label = seq_label[idx]

            tokens, ner_tags = self.getTokenTags(docs[idx])

            if len(text) <= 0 or len(label) <= 0:
                continue

            all_res.append({
                "id": key,
                "text": text,
                "label": label,
                "tokens": tokens,
                "ner_tags": ner_tags,
            })
            
            key += 1

        ids = [r["id"] for r in all_res]

        random.seed(4)
        random.shuffle(ids)
        random.shuffle(ids)
        random.shuffle(ids)
        
        train, validation, test = np.split(ids, [int(len(ids)*0.70), int(len(ids)*0.80)])

        if split == "train":
            allowed_ids = list(train)
        elif split == "validation":
            allowed_ids = list(validation)
        elif split == "test":
            allowed_ids = list(test)
        
        for r in all_res:
            if r["id"] in allowed_ids:
                yield r["id"], r
