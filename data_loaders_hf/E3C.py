# pip install bs4 syntok

import os
import random

import datasets

import numpy as np
from bs4 import BeautifulSoup, ResultSet
from syntok.tokenizer import Tokenizer

tokenizer = Tokenizer()

_CITATION = """\
@report{Magnini2021, \
    author = {Bernardo Magnini and Begoña Altuna and Alberto Lavelli and Manuela Speranza \
    and Roberto Zanoli and Fondazione Bruno Kessler}, \
    keywords = {Clinical data,clinical enti-ties,corpus,multilingual,temporal information}, \
    title = {The E3C Project: \
    European Clinical Case Corpus El proyecto E3C: European Clinical Case Corpus}, \
    url = {https://uts.nlm.nih.gov/uts/umls/home}, \
    year = {2021}, \
}
"""

_DESCRIPTION = """\
E3C is a freely available multilingual corpus (English, French, Italian, Spanish, and Basque) \
of semantically annotated clinical narratives to allow for the linguistic analysis, benchmarking, \
and training of information extraction systems. It consists of two types of annotations: \
(i) clinical entities (e.g., pathologies), (ii) temporal information and factuality (e.g., events). \
Researchers can use the benchmark training and test splits of our corpus to develop and test \
their own models.
"""

_URL = "https://github.com/hltfbk/E3C-Corpus/archive/refs/tags/v2.0.0.zip"

_LANGUAGES = ["English","Spanish","Basque","French","Italian"]

class E3C(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"{lang}_clinical", version="1.0.0", description=f"The {lang} subset of the E3C corpus") for lang in _LANGUAGES
    ]

    BUILDER_CONFIGS += [
        datasets.BuilderConfig(name=f"{lang}_temporal", version="1.0.0", description=f"The {lang} subset of the E3C corpus") for lang in _LANGUAGES
    ]    
    
    DEFAULT_CONFIG_NAME = "French_clinical"

    def _info(self):
        
        if self.config.name == "default":
            self.config.name = self.DEFAULT_CONFIG_NAME

        if self.config.name.find("clinical") != -1:
            names = ["O","B-CLINENTITY","I-CLINENTITY"]
        elif self.config.name.find("temporal") != -1:
            names = ["O", "B-EVENT", "B-ACTOR", "B-BODYPART", "B-TIMEX3", "B-RML", "I-EVENT", "I-ACTOR", "I-BODYPART", "I-TIMEX3", "I-RML"]
        
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=names,
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

        if self.config.name.find("clinical") != -1:
            
            print("clinical")
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_clinical",""), "layer2"),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_clinical",""), "layer2"),
                        "split": "validation",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_clinical",""), "layer1"),
                        "split": "test",
                    },
                ),
            ]
            
        elif self.config.name.find("temporal") != -1:
            
            print("temporal")
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_temporal",""), "layer1"),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_temporal",""), "layer1"),
                        "split": "validation",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "E3C-Corpus-2.0.0/data_annotation", self.config.name.replace("_temporal",""), "layer1"),
                        "split": "test",
                    },
                ),
            ]

    @staticmethod
    def get_annotations(entities: ResultSet, text: str) -> list:

        return [[
            int(entity.get("begin")),
            int(entity.get("end")),
            text[int(entity.get("begin")) : int(entity.get("end"))],
        ] for entity in entities]

    def get_clinical_annotations(self, entities: ResultSet, text: str) -> list:

        return [[
            int(entity.get("begin")),
            int(entity.get("end")),
            text[int(entity.get("begin")) : int(entity.get("end"))],
            entity.get("entityID"),
        ] for entity in entities]

    def get_parsed_data(self, filepath: str):

        for root, _, files in os.walk(filepath):
            
            for file in files:
            
                with open(f"{root}/{file}") as soup_file:
            
                    soup = BeautifulSoup(soup_file, "xml")
                    text = soup.find("cas:Sofa").get("sofaString")
            
                    yield {
                        "CLINENTITY": self.get_clinical_annotations(soup.find_all("custom:CLINENTITY"), text),
                        "EVENT": self.get_annotations(soup.find_all("custom:EVENT"), text),
                        "ACTOR": self.get_annotations(soup.find_all("custom:ACTOR"), text),
                        "BODYPART": self.get_annotations(soup.find_all("custom:BODYPART"), text),
                        "TIMEX3": self.get_annotations(soup.find_all("custom:TIMEX3"), text),
                        "RML": self.get_annotations(soup.find_all("custom:RML"), text),
                        "SENTENCE": self.get_annotations(soup.find_all("type4:Sentence"), text),
                        "TOKENS": self.get_annotations(soup.find_all("type4:Token"), text),
                    }

    def _generate_examples(self, filepath, split):

        all_res = []

        key = 0

        parsed_content = self.get_parsed_data(filepath)

        for content in parsed_content:

            for sentence in content["SENTENCE"]:

                tokens = [(
                    token.offset + sentence[0],
                    token.offset + sentence[0] + len(token.value),
                    token.value,
                ) for token in list(tokenizer.tokenize(sentence[-1]))]

                filtered_tokens = list(
                    filter(
                        lambda token: token[0] >= sentence[0] and token[1] <= sentence[1],
                        tokens,
                    )
                )

                tokens_offsets = [
                    [token[0] - sentence[0], token[1] - sentence[0]] for token in filtered_tokens
                ]

                clinical_labels = ["O"] * len(filtered_tokens)
                clinical_cuid = ["CUI_LESS"] * len(filtered_tokens)
                temporal_information_labels = ["O"] * len(filtered_tokens)

                for entity_type in ["CLINENTITY","EVENT","ACTOR","BODYPART","TIMEX3","RML"]:

                    if len(content[entity_type]) != 0:

                        for entities in list(content[entity_type]):

                            annotated_tokens = [
                                idx_token
                                for idx_token, token in enumerate(filtered_tokens)
                                if token[0] >= entities[0] and token[1] <= entities[1]
                            ]

                            for idx_token in annotated_tokens:

                                if entity_type == "CLINENTITY":
                                    if idx_token == annotated_tokens[0]:
                                        clinical_labels[idx_token] = f"B-{entity_type}"
                                    else:
                                        clinical_labels[idx_token] = f"I-{entity_type}"
                                    clinical_cuid[idx_token] = entities[-1]
                                else:
                                    if idx_token == annotated_tokens[0]:
                                        temporal_information_labels[idx_token] = f"B-{entity_type}"
                                    else:
                                        temporal_information_labels[idx_token] = f"I-{entity_type}"

                if self.config.name.find("clinical") != -1:
                    _labels = clinical_labels        
                elif self.config.name.find("temporal") != -1:
                    _labels = temporal_information_labels
                
                all_res.append({
                    "id": key,
                    "text": sentence[-1],
                    "tokens": list(map(lambda token: token[2], filtered_tokens)),
                    "ner_tags": _labels,
                })
                
                key += 1
        
        if self.config.name.find("clinical") != -1:
            
            if split != "test":
                
                ids = [r["id"] for r in all_res]
        
                random.seed(4)
                random.shuffle(ids)
                random.shuffle(ids)
                random.shuffle(ids)
                
                train, validation = np.split(ids, [int(len(ids)*0.8738)])
        
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
        
        elif self.config.name.find("temporal") != -1:
            
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
