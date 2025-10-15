import os
import re
import ast
import json
import random
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple

import datasets
import numpy as np

_CITATION = """\
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
"""

_DESCRIPTION = """\
ddd
"""

_HOMEPAGE = "ddd"

_LICENSE = "unknown"

_SPECIALITIES = ['immunitaire', 'endocriniennes', 'blessures', 'chimiques', 'etatsosy', 'nutritionnelles', 'infections', 'virales', 'parasitaires', 'tumeur', 'osteomusculaires', 'stomatognathique', 'digestif', 'respiratoire', 'ORL', 'nerveux', 'oeil', 'homme', 'femme', 'cardiovasculaires', 'hemopathies', 'genetique', 'peau']

_LABELS_BASE = ['anatomie', 'date', 'dose', 'duree', 'examen', 'frequence', 'mode', 'moment', 'pathologie', 'sosy', 'substance', 'traitement', 'valeur']


class DEFT2021(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "ner"

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="cls", version="1.0.0", description="DEFT 2021 corpora - Classification task"),
        datasets.BuilderConfig(name="ner", version="1.0.0", description="DEFT 2021 corpora - Named-entity recognition task"),
    ]

    def _info(self):

        if self.config.name.find("cls") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "specialities": datasets.Sequence(
                        datasets.features.ClassLabel(names=_SPECIALITIES),
                    ),
                    "specialities_one_hot": datasets.Sequence(
                        datasets.Value("float"),
                    ),
                }
            )

        elif self.config.name.find("ner") != -1:

            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['O', 'B-anatomie', 'I-anatomie', 'B-date', 'I-date', 'B-dose', 'I-dose', 'B-duree', 'I-duree', 'B-examen', 'I-examen', 'B-frequence', 'I-frequence', 'B-mode', 'I-mode', 'B-moment', 'I-moment', 'B-pathologie', 'I-pathologie', 'B-sosy', 'I-sosy', 'B-substance', 'I-substance', 'B-traitement', 'I-traitement', 'B-valeur', 'I-valeur'],
                        )
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

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")

        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            ),
        ]

    def remove_prefix(self, a: str, prefix: str) -> str:
        if a.startswith(prefix):
            a = a[len(prefix):]
        return a

    def parse_brat_file(self, txt_file: Path, annotation_file_suffixes: List[str] = None, parse_notes: bool = False) -> Dict:

        example = {}
        example["document_id"] = txt_file.with_suffix("").name
        with txt_file.open() as f:
            example["text"] = f.read()

        # If no specific suffixes of the to-be-read annotation files are given - take standard suffixes
        # for event extraction
        if annotation_file_suffixes is None:
            annotation_file_suffixes = [".a1", ".a2", ".ann"]

        if len(annotation_file_suffixes) == 0:
            raise AssertionError(
                "At least one suffix for the to-be-read annotation files should be given!"
            )

        ann_lines = []
        for suffix in annotation_file_suffixes:
            annotation_file = txt_file.with_suffix(suffix)
            if annotation_file.exists():
                with annotation_file.open() as f:
                    ann_lines.extend(f.readlines())

        example["text_bound_annotations"] = []
        example["events"] = []
        example["relations"] = []
        example["equivalences"] = []
        example["attributes"] = []
        example["normalizations"] = []

        if parse_notes:
            example["notes"] = []

        for line in ann_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):  # Text bound
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]
                ann["offsets"] = []
                span_str = self.remove_prefix(fields[1], (ann["type"] + " "))
                text = fields[2]
                for span in span_str.split(";"):
                    start, end = span.split()
                    ann["offsets"].append([int(start), int(end)])

                # Heuristically split text of discontiguous entities into chunks
                ann["text"] = []
                if len(ann["offsets"]) > 1:
                    i = 0
                    for start, end in ann["offsets"]:
                        chunk_len = end - start
                        ann["text"].append(text[i: chunk_len + i])
                        i += chunk_len
                        while i < len(text) and text[i] == " ":
                            i += 1
                else:
                    ann["text"] = [text]

                example["text_bound_annotations"].append(ann)

            elif line.startswith("E"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

                ann["arguments"] = []
                for role_ref_id in fields[1].split()[1:]:
                    argument = {
                        "role": (role_ref_id.split(":"))[0],
                        "ref_id": (role_ref_id.split(":"))[1],
                    }
                    ann["arguments"].append(argument)

                example["events"].append(ann)

            elif line.startswith("R"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]

                ann["head"] = {
                    "role": fields[1].split()[1].split(":")[0],
                    "ref_id": fields[1].split()[1].split(":")[1],
                }
                ann["tail"] = {
                    "role": fields[1].split()[2].split(":")[0],
                    "ref_id": fields[1].split()[2].split(":")[1],
                }

                example["relations"].append(ann)

            # '*' seems to be the legacy way to mark equivalences,
            # but I couldn't find any info on the current way
            # this might have to be adapted dependent on the brat version
            # of the annotation
            elif line.startswith("*"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["ref_ids"] = fields[1].split()[1:]

                example["equivalences"].append(ann)

            elif line.startswith("A") or line.startswith("M"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                info = fields[1].split()
                ann["type"] = info[0]
                ann["ref_id"] = info[1]

                if len(info) > 2:
                    ann["value"] = info[2]
                else:
                    ann["value"] = ""

                example["attributes"].append(ann)

            elif line.startswith("N"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = fields[2]

                info = fields[1].split()

                ann["type"] = info[0]
                ann["ref_id"] = info[1]
                ann["resource_name"] = info[2].split(":")[0]
                ann["cuid"] = info[2].split(":")[1]
                example["normalizations"].append(ann)

            elif parse_notes and line.startswith("#"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = fields[2] if len(fields) == 3 else "<BB_NULL_STR>"

                info = fields[1].split()

                ann["type"] = info[0]
                ann["ref_id"] = info[1]
                example["notes"].append(ann)
        return example

    def _to_source_example(self, brat_example: Dict) -> Dict:

        source_example = {
            "document_id": brat_example["document_id"],
            "text": brat_example["text"],
        }

        source_example["entities"] = []

        for entity_annotation in brat_example["text_bound_annotations"]:
            entity_ann = entity_annotation.copy()

            # Change id property name
            entity_ann["entity_id"] = entity_ann["id"]
            entity_ann.pop("id")

            # Add entity annotation to sample
            source_example["entities"].append(entity_ann)

        return source_example

    def convert_to_prodigy(self, json_object, list_label):

        def prepare_split(text):

            rep_before = ['?', '!', ';', '*']
            rep_after = ['’', "'"]
            rep_both = ['-', '/', '[', ']', ':', ')', '(', ',', '.']

            for i in rep_before:
                text = text.replace(i, ' ' + i)

            for i in rep_after:
                text = text.replace(i, i + ' ')

            for i in rep_both:
                text = text.replace(i, ' ' + i + ' ')

            text_split = text.split()

            punctuations = [',', '.']
            for j in range(0, len(text_split) - 1):
                if j - 1 >= 0 and j + 1 <= len(text_split) - 1 and text_split[j - 1][-1].isdigit() and text_split[j + 1][0].isdigit():
                    if text_split[j] in punctuations:
                        text_split[j - 1:j + 2] = [''.join(text_split[j - 1:j + 2])]

            text = ' '.join(text_split)

            return text

        new_json = []

        for ex in [json_object]:

            text = prepare_split(ex['text'])

            tokenized_text = text.split()

            list_spans = []

            for a in ex['entities']:

                for o in range(len(a['offsets'])):

                    text_annot = prepare_split(a['text'][o])

                    offset_start = a['offsets'][o][0]
                    offset_end = a['offsets'][o][1]

                    nb_tokens_annot = len(text_annot.split())

                    txt_offsetstart = prepare_split(ex['text'][:offset_start])

                    nb_tokens_before_annot = len(txt_offsetstart.split())

                    token_start = nb_tokens_before_annot
                    token_end = token_start + nb_tokens_annot - 1

                    if a['type'] in list_label:
                        list_spans.append({
                            'start': offset_start,
                            'end': offset_end,
                            'token_start': token_start,
                            'token_end': token_end,
                            'label': a['type'],
                            'id': a['entity_id'],
                            'text': a['text'][o],
                        })

            res = {
                'id': ex['document_id'],
                'document_id': ex['document_id'],
                'text': ex['text'],
                'tokens': tokenized_text,
                'spans': list_spans
            }

            new_json.append(res)

        return new_json

    def convert_to_hf_format(self, json_object):

        dict_out = []

        for i in json_object:

            # Filter annotations to keep the longest annotated spans when there is nested annotations
            selected_annotations = []

            if 'spans' in i:

                for idx_j, j in enumerate(i['spans']):

                    len_j = int(j['end']) - int(j['start'])
                    range_j = [l for l in range(int(j['start']), int(j['end']), 1)]

                    keep = True

                    for idx_k, k in enumerate(i['spans'][idx_j + 1:]):

                        len_k = int(k['end']) - int(k['start'])
                        range_k = [l for l in range(int(k['start']), int(k['end']), 1)]

                        inter = list(set(range_k).intersection(set(range_j)))
                        if len(inter) > 0 and len_j < len_k:
                            keep = False

                    if keep:
                        selected_annotations.append(j)

            # Create list of labels + id to separate different annotation and prepare IOB2 format
            nb_tokens = len(i['tokens'])
            ner_tags = ['O'] * nb_tokens

            for slct in selected_annotations:

                for x in range(slct['token_start'], slct['token_end'] + 1, 1):

                    if i['tokens'][x] not in slct['text']:
                        if ner_tags[x - 1] == 'O':
                            ner_tags[x - 1] = slct['label'] + '-' + slct['id']
                    else:
                        if ner_tags[x] == 'O':
                            ner_tags[x] = slct['label'] + '-' + slct['id']

            # Make IOB2 format
            ner_tags_IOB2 = []
            for idx_l, label in enumerate(ner_tags):

                if label == 'O':
                    ner_tags_IOB2.append('O')
                else:
                    current_label = label.split('-')[0]
                    current_id = label.split('-')[1]
                    if idx_l == 0:
                        ner_tags_IOB2.append('B-' + current_label)
                    elif current_label in ner_tags[idx_l - 1]:
                        if current_id == ner_tags[idx_l - 1].split('-')[1]:
                            ner_tags_IOB2.append('I-' + current_label)
                        else:
                            ner_tags_IOB2.append('B-' + current_label)
                    else:
                        ner_tags_IOB2.append('B-' + current_label)

            dict_out.append({
                'id': i['id'],
                'document_id': i['document_id'],
                "ner_tags": ner_tags_IOB2,
                "tokens": i['tokens'],
            })

        return dict_out

    def split_sentences(self, json_o):
        """
            Split each document in sentences to fit the 512 maximum tokens of BERT.

        """

        final_json = []

        for i in json_o:

            ind_punc = [index for index, value in enumerate(i['tokens']) if value == '.'] + [len(i['tokens'])]

            for index, value in enumerate(ind_punc):

                if index == 0:
                    final_json.append({'id': i['id'] + '_' + str(index),
                                    'document_id': i['document_id'],
                                    'ner_tags': i['ner_tags'][:value + 1],
                                    'tokens': i['tokens'][:value + 1]
                                    })
                else:
                    prev_value = ind_punc[index - 1]
                    final_json.append({'id': i['id'] + '_' + str(index),
                                    'document_id': i['document_id'],
                                    'ner_tags': i['ner_tags'][prev_value + 1:value + 1],
                                    'tokens': i['tokens'][prev_value + 1:value + 1]
                                    })

        return final_json

    def _generate_examples(self, data_dir, split):

        if self.config.name.find("cls") != -1:

            all_res = {}

            key = 0

            if split == 'train' or split == 'validation':
                split_eval = 'train'
            else:
                split_eval = 'test'

            path_labels = Path(data_dir) / 'evaluations' / f"ref-{split_eval}-deft2021.txt"

            with open(os.path.join(data_dir, 'distribution-corpus.txt')) as f_dist:

                doc_specialities_ = {}

                with open(path_labels) as f_spec:

                    doc_specialities = [line.strip() for line in f_spec.readlines()]

                    for raw in doc_specialities:

                        raw_split = raw.split('\t')

                        if len(raw_split) == 3 and raw_split[0] in doc_specialities_:
                            doc_specialities_[raw_split[0]].append(raw_split[1])

                        elif len(raw_split) == 3 and raw_split[0] not in doc_specialities_:
                            doc_specialities_[raw_split[0]] = [raw_split[1]]

                ann_path = Path(data_dir) / "DEFT-cas-cliniques"

                for guid, txt_file in enumerate(sorted(ann_path.glob("*.txt"))):

                    ann_file = txt_file.with_suffix("").name.split('.')[0] + '.ann'

                    if ann_file in doc_specialities_:

                        res = {}
                        res['document_id'] = txt_file.with_suffix("").name
                        with txt_file.open() as f:
                            res["text"] = f.read()

                        specialities = doc_specialities_[ann_file]

                        # Empty one hot vector
                        one_hot = [0.0 for i in _SPECIALITIES]

                        # Fill up the one hot vector
                        for s in specialities:
                            one_hot[_SPECIALITIES.index(s)] = 1.0

                        all_res[res['document_id']] = {
                            "id": str(key),
                            "document_id": res['document_id'],
                            "text": res["text"],
                            "specialities": specialities,
                            "specialities_one_hot": one_hot,
                        }

                        key += 1

                distribution = [line.strip() for line in f_dist.readlines()]

                random.seed(4)
                train = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[3] == 'train 2021']
                random.shuffle(train)
                random.shuffle(train)
                random.shuffle(train)
                train, validation = np.split(train, [int(len(train) * 0.7096)])

                test = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[3] == 'test 2021']

                if split == "train":
                    allowed_ids = list(train)
                elif split == "test":
                    allowed_ids = list(test)
                elif split == "validation":
                    allowed_ids = list(validation)

                for r in all_res.values():
                    if r["document_id"] + '.txt' in allowed_ids:
                        yield r["id"], r

        elif self.config.name.find("ner") != -1:

            all_res = []

            key = 0

            with open(os.path.join(data_dir, 'distribution-corpus.txt')) as f_dist:

                distribution = [line.strip() for line in f_dist.readlines()]

                random.seed(4)
                train = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[3] == 'train 2021']
                random.shuffle(train)
                random.shuffle(train)
                random.shuffle(train)
                train, validation = np.split(train, [int(len(train) * 0.73)])
                test = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[3] == 'test 2021']

                ann_path = Path(data_dir) / "DEFT-cas-cliniques"

                for guid, txt_file in enumerate(sorted(ann_path.glob("*.txt"))):

                    brat_example = self.parse_brat_file(txt_file, parse_notes=True)

                    source_example = self._to_source_example(brat_example)

                    prod_format = self.convert_to_prodigy(source_example, _LABELS_BASE)

                    hf_format = self.convert_to_hf_format(prod_format)

                    hf_split = self.split_sentences(hf_format)

                    for h in hf_split:

                        if len(h['tokens']) > 0 and len(h['ner_tags']) > 0:

                            all_res.append({
                                "id": str(key),
                                "document_id": h['document_id'],
                                "tokens": h['tokens'],
                                "ner_tags": h['ner_tags'],
                            })

                            key += 1

                if split == "train":
                    allowed_ids = list(train)
                elif split == "validation":
                    allowed_ids = list(validation)
                elif split == "test":
                    allowed_ids = list(test)

                for r in all_res:
                    if r["document_id"] + '.txt' in allowed_ids:
                        yield r["id"], r
