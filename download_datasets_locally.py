from datasets import load_dataset

ds = [
    ["CLISTER", None],
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
    ["DEFT2019", None],
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


ds = [
    ["QUAERO", "emea"],
    ["QUAERO", "medline"],
]


def save_locally(arr):

    print(arr)

    corpus, subset = arr

    dataset = load_dataset(
        f"Dr-BERT/{corpus}",
        subset,
        data_dir=f"./recipes/{corpus.lower()}/data/",
    )
    dataset.save_to_disk(f"./recipes/{corpus.lower()}/data/local_hf_{subset}/")

for d in ds:
    save_locally(d)
