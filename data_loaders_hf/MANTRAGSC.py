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

# pip install xmltodict

import random
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple

import xmltodict
import numpy as np

import datasets

_CITATION = """\
@article{10.1093/jamia/ocv037,
	author = {Kors, Jan A and Clematide, Simon and Akhondi,
	Saber A and van Mulligen, Erik M and Rebholz-Schuhmann, Dietrich},
	title = "{A multilingual gold-standard corpus for biomedical concept recognition: the Mantra GSC}",
	journal = {Journal of the American Medical Informatics Association},
	volume = {22},
	number = {5},
	pages = {948-956},
	year = {2015},
	month = {05},
	abstract = "{Objective To create a multilingual gold-standard corpus for biomedical concept recognition.Materials
	and methods We selected text units from different parallel corpora (Medline abstract titles, drug labels,
	biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language
	independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and
	covering a wide range of semantic groups. To reduce the annotation workload, automatically generated
	preannotations were provided. Individual annotations were automatically harmonized and then adjudicated, and
	cross-language consistency checks were carried out to arrive at the final annotations.Results The number of final
	annotations was 5530. Inter-annotator agreement scores indicate good agreement (median F-score 0.79), and are
	similar to those between individual annotators and the gold standard. The automatically generated harmonized
	annotation set for each language performed equally well as the best annotator for that language.Discussion The use
	of automatic preannotations, harmonized annotations, and parallel corpora helped to keep the manual annotation
	efforts manageable. The inter-annotator agreement scores provide a reference standard for gauging the performance
	of automatic annotation techniques.Conclusion To our knowledge, this is the first gold-standard corpus for
	biomedical concept recognition in languages other than English. Other distinguishing features are the wide variety
	of semantic groups that are being covered, and the diversity of text genres that were annotated.}",
	issn = {1067-5027},
	doi = {10.1093/jamia/ocv037},
	url = {https://doi.org/10.1093/jamia/ocv037},
	eprint = {https://academic.oup.com/jamia/article-pdf/22/5/948/34146393/ocv037.pdf},
}
"""

_DESCRIPTION = """\
We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims)
in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical
concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups.
"""

_HOMEPAGE = "https://biosemantics.erasmusmc.nl/index.php/resources/mantra-gsc"

_LICENSE = "CC_BY_4p0"

_URL = "https://files.ifi.uzh.ch/cl/mantra/gsc/GSC-v1.1.zip"

_LANGUAGES_2 = {
	"es": "Spanish",
	"fr": "French",
	"de": "German",
	"nl": "Dutch",
	"en": "English",
}

_DATASET_TYPES = {
	"emea": "EMEA",
	"medline": "Medline",
	"patents": "Patent",
}

@dataclass
class DrBenchmarkConfig(datasets.BuilderConfig):
	name: str = None
	version: datasets.Version = None
	description: str = None
	schema: str = None
	subset_id: str = None

class MANTRAGSC(datasets.GeneratorBasedBuilder):

	SOURCE_VERSION = datasets.Version("1.0.0")

	BUILDER_CONFIGS = []

	for language, dataset_type in product(_LANGUAGES_2, _DATASET_TYPES):

		if dataset_type == "patents" and language in ["nl", "es"]:
			continue

		BUILDER_CONFIGS.append(
			DrBenchmarkConfig(
				name=f"{language}_{dataset_type}",
				version=SOURCE_VERSION,
				description=f"Mantra GSC {_LANGUAGES_2[language]} {_DATASET_TYPES[dataset_type]} source schema",
				schema="source",
				subset_id=f"{language}_{_DATASET_TYPES[dataset_type]}",
			)
		)

	DEFAULT_CONFIG_NAME = "fr_medline"

	def _info(self):

		if self.config.name.find("emea") != -1:
			names = ['B-ANAT', 'I-ANAT', 'I-PHEN', 'B-PROC', 'I-CHEM', 'I-PHYS', 'B-DEVI', 'O', 'B-PHYS', 'I-DEVI', 'B-OBJC', 'I-DISO', 'B-PHEN', 'I-LIVB', 'B-DISO', 'B-LIVB', 'B-CHEM', 'I-PROC']
		elif self.config.name.find("medline") != -1:
			names = ['B-ANAT', 'I-ANAT', 'B-PROC', 'I-CHEM', 'I-PHYS', 'B-GEOG', 'B-DEVI', 'O', 'B-PHYS', 'I-LIVB', 'B-OBJC', 'I-DISO', 'I-DEVI', 'B-PHEN', 'B-DISO', 'B-LIVB', 'B-CHEM', 'I-PROC']
		elif self.config.name.find("patents") != -1:
			names = ['B-ANAT', 'I-ANAT', 'B-PROC', 'I-CHEM', 'I-PHYS', 'B-DEVI', 'O', 'I-LIVB', 'B-OBJC', 'I-DISO', 'B-PHEN', 'I-PROC', 'B-DISO', 'I-DEVI', 'B-LIVB', 'B-CHEM', 'B-PHYS']
		
		features = datasets.Features(
			{
				"id": datasets.Value("string"),
				"tokens": [datasets.Value("string")],
				"ner_tags": datasets.Sequence(
					datasets.features.ClassLabel(
						names = names,
					)
				),
			}
		)

		return datasets.DatasetInfo(
			description=_DESCRIPTION,
			features=features,
			homepage=_HOMEPAGE,
			license=str(_LICENSE),
			citation=_CITATION,
		)

	def _split_generators(self, dl_manager):

		language, dataset_type = self.config.name.split("_")

		data_dir = dl_manager.download_and_extract(_URL)
		data_dir = Path(data_dir) / "GSC-v1.1" / f"{_DATASET_TYPES[dataset_type]}_GSC_{language}_man.xml"

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

	def _generate_examples(self, data_dir, split):

		with open(data_dir) as fd:
			doc = xmltodict.parse(fd.read())

		all_res = []

		for d in doc["Corpus"]["document"]:

			if type(d["unit"]) != type(list()):
				d["unit"] = [d["unit"]]

			for u in d["unit"]:

				text = u["text"]

				if "e" in u.keys():

					if type(u["e"]) != type(list()):
						u["e"] = [u["e"]]
					
					tags = [{
						"label": current["@grp"].upper(),
						"offset_start": int(current["@offset"]),
						"offset_end": int(current["@offset"]) + int(current["@len"]),
					} for current in u["e"]]

				else:
					tags = []

				_tokens = text.split(" ")
				tokens = []
				for i, t in enumerate(_tokens):

					concat = " ".join(_tokens[0:i+1])

					offset_start = len(concat) - len(t)
					offset_end = len(concat)

					tokens.append({
						"token": t,
						"offset_start": offset_start,
						"offset_end": offset_end,
					})

				ner_tags = [["O", 0] for o in tokens]

				for tag in tags:

					cpt = 0

					for idx, token in enumerate(tokens):

						rtok = range(token["offset_start"], token["offset_end"]+1)
						rtag = range(tag["offset_start"], tag["offset_end"]+1)

						# Check if the ranges are overlapping
						if bool(set(rtok) & set(rtag)):

							# if ner_tags[idx] != "O" and ner_tags[idx] != tag['label']:
							# 	print(f"{token} - currently: {ner_tags[idx]} - after: {tag['label']}")
							
							if ner_tags[idx][0] == "O":
								cpt += 1
								ner_tags[idx][0] = tag["label"]
								ner_tags[idx][1] = cpt

				for i in range(len(ner_tags)):

					tag = ner_tags[i][0]

					if tag == "O":
						continue
					elif tag != "O" and ner_tags[i][1] == 1:
						ner_tags[i][0] = "B-" + tag
					elif tag != "O" and ner_tags[i][1] != 1:
						ner_tags[i][0] = "I-" + tag

				obj = {
					"id": u["@id"],
					"tokens": [t["token"] for t in tokens],
					"ner_tags": [n[0] for n in ner_tags],
				}

				all_res.append(obj)
		
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
			identifier = r["id"]
			if identifier in allowed_ids:
				yield identifier, r 
