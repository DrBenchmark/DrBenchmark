import os
import aiohttp
import logging

from datasets import load_dataset

# Root of the repository: the datasets are saved in the recipes, wherever the script is run from
REPO_DIR = os.path.dirname(os.path.realpath(__file__))


ds = [
    ["CLISTER", "source"],
    ["DEFT2020", "task_1"],
    ["DEFT2020", "task_2"],
    ["E3C", "French_clinical"],
    ["E3C", "French_temporal"],
    ["FrenchMedMCQA", None],
    ["MANTRAGSC", "fr_emea"],
    ["MANTRAGSC", "fr_medline"],
    ["MANTRAGSC", "fr_patents"],
    ["MORFITT", "source"],
    ["QUAERO", "emea"],
    ["QUAERO", "medline"],
    ["PxCorpus", None],
    ["DiaMED", None],
    # ["DEFT2019", None],
    ["DEFT2021", "cls"],
    ["DEFT2021", "ner"],

    ["CAS", "pos"],
    ["CAS", "cls"],
    ["CAS", "ner_neg"],
    ["CAS", "ner_spec"],

    ["ESSAI", "pos"],
    ["ESSAI", "cls"],
    ["ESSAI", "ner_neg"],
    ["ESSAI", "ner_spec"],
]


def save_locally(arr):
    corpus, subset = arr

    logging.info(f">> Downloading {corpus}{' - '+subset if subset else ''}")

    dataset = load_dataset(
        f"DrBenchmark/{corpus}",
        subset,
        trust_remote_code=True,
        storage_options={
            'client_kwargs': {'timeout': aiohttp.ClientTimeout(20 * 60)}
        }
    )
    dataset.save_to_disk(os.path.join(REPO_DIR, "recipes", corpus.lower(), "data", f"local_hf_{subset}"))


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    for d in ds:
        save_locally(d)
