# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import re
import os
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
ddd
"""

_HOMEPAGE = "http://natalia.grabar.free.fr/resources.php#cas"

_LICENSE = "ddd"

_LABELS_BASE = ['age', 'genre', 'issue', 'origine']

@dataclass
class DrBenchmarkConfig(datasets.BuilderConfig):
	name: str = None
	version: datasets.Version = None
	description: str = None
	schema: str = None
	subset_id: str = None

class DEFT2019(datasets.GeneratorBasedBuilder):

	SOURCE_VERSION = datasets.Version("1.0.0")

	BUILDER_CONFIGS = []

	BUILDER_CONFIGS.append(
		DrBenchmarkConfig(
			name=f"DEFT_2019",
			version=SOURCE_VERSION,
			description=f"DEFT 2019",
			schema="source"
		)
	)

	def _info(self) -> datasets.DatasetInfo:

		return datasets.DatasetInfo(
			description=_DESCRIPTION,
			features=datasets.Features(
				{
					"id": datasets.Value("string"),
					"document_id": datasets.Value("string"),
					"tokens": datasets.Sequence(datasets.Value("string")),
					"ner_tags": datasets.Sequence(
						datasets.features.ClassLabel(
							names = ['O', 'B-age', 'I-age', 'B-genre', 'I-genre', 'B-issue', 'I-issue', 'B-origine', 'I-origine'],
						)
					),
				}
			),
			supervised_keys=None,
			homepage=_HOMEPAGE,
			citation=_CITATION,
			license=_LICENSE,
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
			a = a[len(prefix) :]
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
						ann["text"].append(text[i : chunk_len + i])
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
				text = text.replace(i, ' '+i)

			for i in rep_after:
				text = text.replace(i, i+' ')

			for i in rep_both:
				text = text.replace(i, ' '+i+' ')

			text_split = text.split()

			punctuations = [',', '.']
			for j in range(0, len(text_split)-1):
				if j-1 >= 0 and j+1 <= len(text_split)-1 and text_split[j-1][-1].isdigit() and text_split[j+1][0].isdigit():
					if text_split[j] in punctuations:
						text_split[j-1:j+2] = [''.join(text_split[j-1:j+2])]

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

					len_j = int(j['end'])-int(j['start'])
					range_j = [l for l in range(int(j['start']),int(j['end']),1)]

					keep = True

					for idx_k, k in enumerate(i['spans'][idx_j+1:]):

						len_k = int(k['end'])-int(k['start'])
						range_k = [l for l in range(int(k['start']),int(k['end']),1)]

						inter = list(set(range_k).intersection(set(range_j)))
						if len(inter) > 0 and len_j < len_k:
							keep = False

					if keep:
						selected_annotations.append(j)

			# Create list of labels + id to separate different annotation and prepare IOB2 format
			nb_tokens = len(i['tokens'])
			ner_tags = ['O']*nb_tokens
			
			for slct in selected_annotations:

				for x in range(slct['token_start'], slct['token_end']+1, 1):

					if i['tokens'][x] not in slct['text']:
						if ner_tags[x-1] == 'O':
							ner_tags[x-1] = slct['label']+'-'+slct['id']
					else:
						if ner_tags[x] == 'O':
							ner_tags[x] = slct['label']+'-'+slct['id']

			# Make IOB2 format
			ner_tags_IOB2 = []
			for idx_l, label in enumerate(ner_tags):

				if label == 'O':
					ner_tags_IOB2.append('O')
				else:
					current_label = label.split('-')[0]
					current_id = label.split('-')[1]
					if idx_l == 0:
						ner_tags_IOB2.append('B-'+current_label)
					elif current_label in ner_tags[idx_l-1]:
						if current_id == ner_tags[idx_l-1].split('-')[1]:
							ner_tags_IOB2.append('I-'+current_label)
						else:
							ner_tags_IOB2.append('B-'+current_label)
					else:
						ner_tags_IOB2.append('B-'+current_label)

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

			ind_punc = [index for index, value in enumerate(i['tokens']) if value=='.'] + [len(i['tokens'])]
			
			for index, value in enumerate(ind_punc):
				
				if index==0:
					final_json.append({'id': i['id']+'_'+str(index),
									'document_id': i['document_id'],
									'ner_tags': i['ner_tags'][:value+1],
									'tokens': i['tokens'][:value+1]
									})
				else:
					prev_value = ind_punc[index-1]
					final_json.append({'id': i['id']+'_'+str(index),
									'document_id': i['document_id'],
									'ner_tags': i['ner_tags'][prev_value+1:value+1],
									'tokens': i['tokens'][prev_value+1:value+1]
									}) 
		
		return final_json

	def _generate_examples(self, data_dir, split):
		"""Yields examples as (key, example) tuples."""

		all_res = []

		key = 0

		with open(os.path.join(data_dir, 'distribution-corpus.txt')) as f_dist:

			distribution = [line.strip() for line in f_dist.readlines()]

			random.seed(4)
			train = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[1] == 'train 2019']
			random.shuffle(train)
			random.shuffle(train)
			random.shuffle(train)
			train, validation = np.split(train, [int(len(train)*0.7096)])
			test = [raw.split('\t')[0] for raw in distribution if len(raw.split('\t')) == 4 and raw.split('\t')[1] == 'test 2019']

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
			elif split == "test":
				allowed_ids = list(test)
			elif split == "validation":
				allowed_ids = list(validation)

			for r in all_res:
				if r["document_id"]+'.txt' in allowed_ids:
					yield r["id"], r

