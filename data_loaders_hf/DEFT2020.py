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
"""DEFT 2020 - DEFT (DÉfi Fouille de Textes)"""

import os
import random

import xmltodict
import numpy as np

import datasets

_DESCRIPTION = """\
Dans la continuité de DEFT 2019, l'édition 2020 du défi fouille de textes (DEFT 2020) continue d'explorer les cas cliniques rédigés en français. Cette nouvelle édition porte sur l'extraction d'information fine autour d'une douzaine de catégories (à l'image des campagnes internationales i2b2 2009, 2012 et 2014, ou SemEval 2014). En dehors du domaine clinique, nous proposons également deux nouvelles tâches sur la similarité sémantique entre phrases.

Informations globales sur le corpus
L'un des corpus du défi provient d'un ensemble plus vaste composé de cas cliniques, porteur d'annotations plus complètes [1]. Les cas cliniques couvrent différentes spécialités médicales (cardiologie, urologie, oncologie, obstétrique, pulmonaire, gasto-entérologie, etc.). Ils décrivent des cas qui se sont produits dans différents pays francophones (France, Belgique, Suisse, Canada, pays africains, pays tropicaux, etc.). Le deuxième corpus utilisé provient du projet CLEAR [2]. Le projet CLEAR se compose de trois sous-corpus (articles d'encyclopédie, notices de médicaments, et résumés Cochrane) dont le contenu est comparable. Chaque corpus fournit des versions techniques et simplifiées sur un sujet donné en français. Les phrases proposées dans les tâches 1 et 2 proviennent de ce corpus. Les annotations de référence ont fait l'objet d'un consensus après une double annotation indépendante.

[1] N Grabar, V Claveau, C Dalloux. CAS: French Corpus with Clinical Cases. LOUHI 2018, p. 1-7

[2] N Grabar, R Cardon. CLEAR -- Simple Corpus for Medical French. ATA 2018, p 1-7

Accès aux données
L'accès aux données ne sera rendu possible qu'après signature d'un accord d'utilisation des données DEFT 2020 par l'ensemble des membres de l'équipe. Les participants sont libres de participer à une ou plusieurs tâches. En accédant aux données, ils s'engagent moralement à participer jusqu'au bout (soumettre des résultats et présenter les résultats pendant l'atelier).

Accord d'utilisation des données DEFT 2020 : compléter et faire signer cet accord par tous les membres de l'équipe participante, puis renvoyer cet accord sous forme d'une numérisation au format PDF par mail à deft2020 A@T limsi.fr
Accès aux données hors challenge (DEFT 2019, 2020) : corpus de 717 cas cliniques rédigés en français, annotés avec quatre type d'information démographique (âge, genre) et clinique (origine de la consultation, issue), et pour un sous-ensemble de 167 cas, également annotés avec treize types d'information clinique (anatomie, date, dose, durée, examen, fréquence, mode, moment, pathologie, signe ou symptôme, substance, traitement, valeur) et cinq attributs (assertion, changement, état, norme, prise). La demande d'accès au corpus doit être formulée auprès de Natalia Grabar (natalia.grabar@univ-lille.fr) et Cyril Grouin (cyril.grouin@limsi.fr)
"""

_HOMEPAGE = "https://deft.lisn.upsaclay.fr/2020/"

_LICENSE = "other"

_CITATION = """\
@inproceedings{cardon-etal-2020-presentation,
    title = "Pr{\'e}sentation de la campagne d{'}{\'e}valuation {DEFT} 2020 : similarit{\'e} textuelle en domaine ouvert et extraction d{'}information pr{\'e}cise dans des cas cliniques (Presentation of the {DEFT} 2020 Challenge : open domain textual similarity and precise information extraction from clinical cases )",
    author = "Cardon, R{\'e}mi  and
      Grabar, Natalia  and
      Grouin, Cyril  and
      Hamon, Thierry",
    booktitle = "Actes de la 6e conf{\'e}rence conjointe Journ{\'e}es d'{\'E}tudes sur la Parole (JEP, 33e {\'e}dition), Traitement Automatique des Langues Naturelles (TALN, 27e {\'e}dition), Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (R{\'E}CITAL, 22e {\'e}dition). Atelier D{\'E}fi Fouille de Textes",
    month = "6",
    year = "2020",
    address = "Nancy, France",
    publisher = "ATALA et AFCP",
    url = "https://aclanthology.org/2020.jeptalnrecital-deft.1",
    pages = "1--13",
    abstract = "L{'}{\'e}dition 2020 du d{\'e}fi fouille de texte (DEFT) a propos{\'e} deux t{\^a}ches autour de la similarit{\'e} textuelle et une t{\^a}che d{'}extraction d{'}information. La premi{\`e}re t{\^a}che vise {\`a} identifier le degr{\'e} de similarit{\'e} entre paires de phrases sur une {\'e}chelle de 0 (le moins similaire) {\`a} 5 (le plus similaire). Les r{\'e}sultats varient de 0,65 {\`a} 0,82 d{'}EDRM. La deuxi{\`e}me t{\^a}che consiste {\`a} d{\'e}terminer la phrase la plus proche d{'}une phrase source parmi trois phrases cibles fournies, avec des r{\'e}sultats tr{\`e}s {\'e}lev{\'e}s, variant de 0,94 {\`a} 0,99 de pr{\'e}cision. Ces deux t{\^a}ches reposent sur un corpus du domaine g{\'e}n{\'e}ral et de sant{\'e}. La troisi{\`e}me t{\^a}che propose d{'}extraire dix cat{\'e}gories d{'}informations du domaine m{\'e}dical depuis le corpus de cas cliniques de DEFT 2019. Les r{\'e}sultats varient de 0,07 {\`a} 0,66 de F-mesure globale pour la sous-t{\^a}che des pathologies et signes ou sympt{\^o}mes, et de 0,14 {\`a} 0,76 pour la sous-t{\^a}che sur huit cat{\'e}gories m{\'e}dicales. Les m{\'e}thodes utilis{\'e}es reposent sur des CRF et des r{\'e}seaux de neurones.",
    language = "French",
}
"""


class FrenchMedMCQA(datasets.GeneratorBasedBuilder):
    """DEFT 2020 - DEFT (DÉfi Fouille de Textes)"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="task_1", version=VERSION, description="Tâche 1 : identifier le degré de similarité entre paires de phrases parallèles et non-parallèles sur plusieurs domaines"),
        datasets.BuilderConfig(name="task_2", version=VERSION, description="Tâche 2 : identifier les phrases parallèles possible pour une phrase source"),
    ]

    DEFAULT_CONFIG_NAME = "task_1"

    def _info(self):

        if self.config.name == "default":
            self.config.name = self.DEFAULT_CONFIG_NAME

        if self.config.name == "task_1":

            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "moy": datasets.Value("float"),
                        "vote": datasets.Value("float"),
                        "scores": datasets.Sequence(datasets.Value("float")),
                        "source": datasets.Value("string"),
                        "cible": datasets.Value("string"),
                    }
                ),
                supervised_keys=None,
                homepage="https://deft.lisn.upsaclay.fr/2020/",
                citation=_CITATION,
                license=_LICENSE,
            )

        elif self.config.name == "task_2":

            return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "correct_cible": datasets.features.ClassLabel(
                            names=['1', '2', '3'],
                        ),
                        "source": datasets.Value("string"),
                        "cible_1": datasets.Value("string"),
                        "cible_2": datasets.Value("string"),
                        "cible_3": datasets.Value("string"),
                    }
                ),
                supervised_keys=None,
                homepage="https://deft.lisn.upsaclay.fr/2020/",
                citation=_CITATION,
                license=_LICENSE,
            )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        data_dir = self.config.data_dir

        if self.config.name == "task_1":

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t1-train.xml"),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t1-train.xml"),
                        "split": "validation",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t1-test.xml"),
                        "split": "test",
                    },
                ),
            ]

        elif self.config.name == "task_2":

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t2-train.xml"),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t2-train.xml"),
                        "split": "validation",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "t2-test.xml"),
                        "split": "test",
                    },
                ),
            ]

    def _generate_examples(self, filepath, split):

        all_res = []

        if self.config.name == "task_1":

            ratio = 0.8316

            with open(filepath) as fd:
                doc = xmltodict.parse(fd.read())

            for d in doc["doc"]["paire"]:

                if int(d["@id"]) > -1:

                    all_res.append({
                        "id": d["@id"],
                        "vote": d["@vote"],
                        "moy": d["@moy"],
                        "scores": [float(a) for a in d["@scores"].split(",")],
                        "source": d["source"],
                        "cible": d["cible"],
                    })

        elif self.config.name == "task_2":

            ratio = 0.8059

            with open(filepath) as fd:
                doc = xmltodict.parse(fd.read())

            for d in doc["doc"]["ensemble"]:

                obj = {
                    "id": d["@id"],
                    "correct_cible": str(d["@cible"]),
                    "source": d["source"],
                    "cible_1": None,
                    "cible_2": None,
                    "cible_3": None,
                }

                for t in d["cible"]:
                    obj[f"cible_{t['@num']}"] = t["#text"]

                all_res.append(obj)

        if split != "test":

            ids = [r["id"] for r in all_res]

            random.seed(4)
            random.shuffle(ids)
            random.shuffle(ids)
            random.shuffle(ids)

            train, validation = np.split(ids, [int(len(ids) * ratio)])

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
