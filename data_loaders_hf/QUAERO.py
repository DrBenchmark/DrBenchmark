# coding=utf-8

"""QUAERO Corpus"""

import os
import datasets
from tqdm import tqdm

from datasets import load_dataset

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@InProceedings{neveol14quaero, 
  author = {Névéol, Aurélie and Grouin, Cyril and Leixa, Jeremy 
			and Rosset, Sophie and Zweigenbaum, Pierre},
  title = {The {QUAERO} {French} Medical Corpus: A Ressource for
		   Medical Entity Recognition and Normalization}, 
  OPTbooktitle = {Proceedings of the Fourth Workshop on Building 
				 and Evaluating Ressources for Health and Biomedical 
				 Text Processing}, 
  booktitle = {Proc of BioTextMining Work}, 
  OPTseries = {BioTxtM 2014}, 
  year = {2014}, 
  pages = {24--30}, 
}
"""

_LICENSE = """
GNU Free Documentation License v1.3
"""

_DESCRIPTION = """
The QUAERO French Medical Corpus has been initially developed as a resource for named entity recognition and normalization [1]. It was then improved with the purpose of creating a gold standard set of normalized entities for French biomedical text, that was used in the CLEF eHealth evaluation lab [2][3].
A selection of MEDLINE titles and EMEA documents were manually annotated. The annotation process was guided by concepts in the Unified Medical Language System (UMLS):
1. Ten types of clinical entities, as defined by the following UMLS Semantic Groups (Bodenreider and McCray 2003) were annotated: Anatomy, Chemical and Drugs, Devices, Disorders, Geographic Areas, Living Beings, Objects, Phenomena, Physiology, Procedures.
2. The annotations were made in a comprehensive fashion, so that nested entities were marked, and entities could be mapped to more than one UMLS concept. In particular: (a) If a mention can refer to more than one Semantic Group, all the relevant Semantic Groups should be annotated. For instance, the mention “récidive” (recurrence) in the phrase “prévention des récidives” (recurrence prevention) should be annotated with the category “DISORDER” (CUI C2825055) and the category “PHENOMENON” (CUI C0034897); (b) If a mention can refer to more than one UMLS concept within the same Semantic Group, all the relevant concepts should be annotated. For instance, the mention “maniaques” (obsessive) in the phrase “patients maniaques” (obsessive patients) should be annotated with CUIs C0564408 and C0338831 (category “DISORDER”); (c) Entities which span overlaps with that of another entity should still be annotated. For instance, in the phrase “infarctus du myocarde” (myocardial infarction), the mention “myocarde” (myocardium) should be annotated with category “ANATOMY” (CUI C0027061) and the mention “infarctus du myocarde” should be annotated with category “DISORDER” (CUI C0027051)
The QUAERO French Medical Corpus BioC release comprises a subset of the QUAERO French Medical corpus, as follows:
Training data (BRAT version used in CLEF eHealth 2015 task 1b as training data): 
- MEDLINE_train_bioc file: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA_train_bioc file: 3 EMEA documents, segmented into 11 sub-documents, annotated with normalized entities in the BioC format 
Development data  (BRAT version used in CLEF eHealth 2015 task 1b as test data and in CLEF eHealth 2016 task 2 as development data): 
- MEDLINE_dev_bioc file: 832 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA_dev_bioc file: 3 EMEA documents, segmented into 12 sub-documents, annotated with normalized entities in the BioC format 
Test data (BRAT version used in CLEF eHealth 2016 task 2 as test data): 
- MEDLINE_test_bioc folder: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA folder_test_bioc: 4 EMEA documents, segmented into 15 sub-documents, annotated with normalized entities in the BioC format 
This release of the QUAERO French medical corpus, BioC version, comes in the BioC format, through automatic conversion from the original BRAT format obtained with the Brat2BioC tool https://bitbucket.org/nicta_biomed/brat2bioc developped by Jimeno Yepes et al.
Antonio Jimeno Yepes, Mariana Neves, Karin Verspoor 
Brat2BioC: conversion tool between brat and BioC
BioCreative IV track 1 - BioC: The BioCreative Interoperability Initiative, 2013
Please note that the original version of the QUAERO corpus distributed in the CLEF eHealth challenge 2015 and 2016 came in the BRAT stand alone format. It was distributed with the CLEF eHealth evaluation tool. This original distribution of the QUAERO French Medical corpus is available separately from https://quaerofrenchmed.limsi.fr  
All questions regarding the task or data should be addressed to aurelie.neveol@limsi.fr
"""
	
_LABELS_BASE = ['DISO', 'DEVI', 'CHEM', 'GEOG', 'OBJC', 'PHEN', 'PHYS', 'LIVB', 'PROC', 'ANAT']

class QUAERO(datasets.GeneratorBasedBuilder):
	"""QUAERO dataset."""

	VERSION = datasets.Version("1.0.0")

	BUILDER_CONFIGS = [
		datasets.BuilderConfig(name="emea", version=VERSION, description="The EMEA QUAERO corpora"),
		datasets.BuilderConfig(name="medline", version=VERSION, description="The MEDLINE QUAERO corpora"),
	]

	DEFAULT_CONFIG_NAME = "emea"

	def _info(self):

		if self.config.name == "emea":
			
			return datasets.DatasetInfo(
				description=_DESCRIPTION,
				features=datasets.Features(
					{
						"id": datasets.Value("string"),
						"document_id": datasets.Value("string"),
						"tokens": datasets.Sequence(datasets.Value("string")),
						"ner_tags": datasets.Sequence(
							datasets.features.ClassLabel(
								names = ['O', 'B-LIVB', 'I-LIVB', 'B-PROC', 'I-PROC', 'B-ANAT', 'I-ANAT', 'B-DEVI', 'I-DEVI', 'B-CHEM', 'I-CHEM', 'B-GEOG', 'I-GEOG', 'B-PHYS', 'I-PHYS', 'B-PHEN', 'I-PHEN', 'B-DISO', 'I-DISO', 'B-OBJC', 'I-OBJC'],
							)
						),
					}
				),
				supervised_keys=None,
				homepage="https://quaerofrenchmed.limsi.fr/",
				citation=_CITATION,
				license=_LICENSE,
			)
		
		elif self.config.name == "medline":
			return datasets.DatasetInfo(
				description=_DESCRIPTION,
				features=datasets.Features(
					{
						"id": datasets.Value("string"),
						"document_id": datasets.Value("string"),
						"tokens": datasets.Sequence(datasets.Value("string")),
						"ner_tags": datasets.Sequence(
							datasets.features.ClassLabel(
								names = ['O', 'B-LIVB', 'I-LIVB', 'B-PROC', 'I-PROC', 'B-ANAT', 'I-ANAT', 'B-DEVI', 'I-DEVI', 'B-CHEM', 'I-CHEM', 'B-GEOG', 'I-GEOG', 'B-PHYS', 'I-PHYS', 'B-PHEN', 'I-PHEN', 'B-DISO', 'I-DISO', 'B-OBJC', 'I-OBJC'],
							)
						),
					}
				),
				supervised_keys=None,
				homepage="https://quaerofrenchmed.limsi.fr/",
				citation=_CITATION,
				license=_LICENSE,
			)

	def _split_generators(self, dl_manager):

		return [
			datasets.SplitGenerator(
				name=datasets.Split.TRAIN,
				gen_kwargs={
					"split": "train",
				}
			),
			datasets.SplitGenerator(
				name=datasets.Split.VALIDATION,
				gen_kwargs={
					"split": "validation",
				}
			),
			datasets.SplitGenerator(
				name=datasets.Split.TEST,
				gen_kwargs={
					"split": "test",
				}
			),
		]

	def split_sentences(self, json_o):
		"""
			Split le corpus en phrase plus courtes pour que ça fit dans des modèles types BERT

			Le split est fait sur les points "."

		"""
		
		final_json = []
		
		for i in json_o:
			
			ind_punc = [index for index, value in enumerate(i['tokens']) if value=='.'] + [len(i['tokens'])]
			# ind_punc = [index for index, value in enumerate(i['tokens']) if value=='.' and not str(i['tokens'][index-1]).isnumeric()]
			
			for index, value in enumerate(ind_punc):
				
				if index==0:
					final_json.append({'id': i['id']+'_'+str(index),
									'document_id': i['id']+'_'+str(index),
									'ner_tags': i['ner_tags'][:value+1],
									'tokens': i['tokens'][:value+1]
									})
				else:
					prev_value = ind_punc[index-1]
					final_json.append({'id': i['id']+'_'+str(index),
									'document_id': i['document_id']+'_'+str(index),
									'ner_tags': i['ner_tags'][prev_value+1:value+1],
									'tokens': i['tokens'][prev_value+1:value+1]
									}) 
		
		return final_json

	def convert_to_prodigy(self, json_object):
		
		new_json = []

		for ex in json_object:

			tokenized_text = ex['text'].split()

			list_spans = []

			for a in ex['text_bound_annotations']:

				for o in range(len(a['offsets'])):

					offset_start = a['offsets'][o][0]
					offset_end = a['offsets'][o][1]

					nb_tokens_annot = len(a['text'][o].split())

					nb_tokens_before_annot = len(ex['text'][:offset_start].split())
					nb_tokens_after_annot = len(ex['text'][offset_end:].split())

					token_start = nb_tokens_before_annot
					token_end = token_start + nb_tokens_annot - 1

					list_spans.append({
						'start': offset_start,
						'end': offset_end,
						'token_start': token_start,
						'token_end': token_end,
						'label': a['type'],
						'id': a['id'],
						'text': a['text'][o],
					})

			res = {
				'id': ex['id'],
				'document_id': ex['document_id'],
				'text': ex['text'],
				'tokens': tokenized_text,
				'spans': list_spans
			}

			new_json.append(res)
			
		return new_json

	def convert_to_hf_format(self, json_object, list_label):
		"""
		Le format prends en compte le multilabel en faisant une concaténation avec "_" entre chaque label
		"""
		
		dict_out = []
		
		for i in json_object:

			# Filter annotations to keep the longest annotated spans when there is nested annotations
			selected_annotations = []

			if 'spans' in i:

				# print(len(i['spans']))

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

					if slct['label'] in list_label:

						if ner_tags[x] == 'O':
							ner_tags[x] = slct['label']+'-'+slct['id']

			# Make IOB2 format
			ner_tags_IOB2 = []
			for idx_l, label in enumerate(ner_tags):
				# print(label)

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

			# print(ner_tags_IOB2)
			dict_out.append({
				'id': i['id'],
				'document_id': i['document_id'],
				"ner_tags": ner_tags_IOB2,
				"tokens": i['tokens'],
			})
		
		return dict_out

	def _generate_examples(self, split):

		ds = load_dataset("bigbio/quaero", f"quaero_{self.config.name}_source")[split]

		if self.config.name == "emea":
			
			ds = self.split_sentences(
				self.convert_to_hf_format(
					self.convert_to_prodigy(ds),
					_LABELS_BASE,
				)
			)
			
		else:
				
			ds = self.convert_to_hf_format(
				self.convert_to_prodigy(ds),
				_LABELS_BASE,
			)

		for d in ds:
			yield d["id"], d