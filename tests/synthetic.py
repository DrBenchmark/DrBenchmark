"""Tiny synthetic datasets with the same schema as the datasets of the benchmark.

They let the recipes run end to end without network access and without the real data.
Only the columns read by the training scripts are generated. The label names follow the
style of the real ones (IOB2 tags, `B_scope_neg`, ICD-10 chapters, ...), but most label
sets are shortened.
"""

import functools

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

WORDS = (
    "le patient présente une douleur thoracique aiguë depuis trois jours sans fièvre ni toux "
    "avec dyspnée modérée traitement par aspirine et repos examen clinique normal"
).split()

# Number of examples of each split. 12 training examples keep the arithmetic of the
# gradient accumulation tests simple (see tests/test_recipes.py).
SPLIT_SIZES = {"train": 12, "validation": 6, "test": 6}


def sentence(i):
    """Deterministic sentence of 6 to 10 words."""
    start = i % len(WORDS)
    words = WORDS[start:] + WORDS[:start]
    return words[:6 + i % 5]


def text(i):
    return " ".join(sentence(i))


def tags(i, n_labels):
    """One tag per word of `sentence(i)`, cycling through every label."""
    return [(i + j) % n_labels for j in range(len(sentence(i)))]


def one_hot(i, n_labels):
    """One or two active labels per example."""
    active = {i % n_labels}
    if i % 2:
        active.add((i + 1) % n_labels)
    return sorted(active)


def build(columns, features):
    """Build the three splits of a dataset from column generators.

    `columns` maps a column name to a function of the example index.
    """
    splits = {}
    offset = 0
    for split, size in SPLIT_SIZES.items():
        indices = range(offset, offset + size)
        splits[split] = Dataset.from_dict(
            {name: [make(i) for i in indices] for name, make in columns.items()},
            features=Features(features),
        )
        offset += size
    return DatasetDict(splits)


def token_classification(labels, tag_column="ner_tags", extra_columns=None, extra_features=None):
    columns = {
        "id": str,
        "tokens": sentence,
        tag_column: lambda i: tags(i, len(labels)),
    }
    features = {
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        tag_column: Sequence(ClassLabel(names=labels)),
    }
    columns.update(extra_columns or {})
    features.update(extra_features or {})
    return build(columns, features)


def multi_label(labels, text_column):
    def one_hot_vector(i):
        active = one_hot(i, len(labels))
        return [float(j in active) for j in range(len(labels))]

    return build(
        {
            "id": str,
            text_column: text,
            "specialities": lambda i: one_hot(i, len(labels)),
            "specialities_one_hot": one_hot_vector,
        },
        {
            "id": Value("string"),
            text_column: Value("string"),
            "specialities": Sequence(ClassLabel(names=labels)),
            "specialities_one_hot": Sequence(Value("float32")),
        },
    )


POS_TAGS = ["B-ADJ", "B-DET:ART", "B-NOM", "B-PRP", "B-PUN", "B-SENT", "B-VER:", "B-VER:pres"]
NEGATION_CLASSES = ["negation_speculation", "negation", "neutral", "speculation"]
UMLS_TAGS = ["O", "B-ANAT", "I-ANAT", "B-CHEM", "I-CHEM", "B-DISO", "I-DISO", "B-PROC", "I-PROC"]


def cas_like(negation_tags, speculation_tags):
    """CAS and ESSAI share their schema: only the scope labels differ."""
    return {
        "pos": token_classification(POS_TAGS, tag_column="pos_tags"),
        "cls": build(
            {"id": str, "tokens": sentence, "label": lambda i: i % len(NEGATION_CLASSES)},
            {"id": Value("string"), "tokens": Sequence(Value("string")), "label": ClassLabel(names=NEGATION_CLASSES)},
        ),
        "ner_neg": token_classification(negation_tags),
        "ner_spec": token_classification(speculation_tags),
    }


def french_med_mcqa():
    answers = [[0], [1, 2], [0, 3, 4], [4], [0, 1, 2, 3], [0, 1, 2, 3, 4], [2, 3]]
    letters = ["a", "b", "c", "d", "e"]
    columns = {
        "id": str,
        "question": text,
        "correct_answers": lambda i: answers[i % len(answers)],
        "number_correct_answers": lambda i: len(answers[i % len(answers)]) - 1,
    }
    features = {
        "id": Value("string"),
        "question": Value("string"),
        "correct_answers": Sequence(ClassLabel(names=letters)),
        "number_correct_answers": ClassLabel(names=["1", "2", "3", "4", "5"]),
    }
    for rank, letter in enumerate(letters):
        columns[f"answer_{letter}"] = lambda i, rank=rank: text(i + rank + 1)
        features[f"answer_{letter}"] = Value("string")
    return build(columns, features)


def px_corpus():
    intents = ["medical_prescription", "negate", "none", "replace"]
    return token_classification(
        ["O", "B-A", "B-drug", "I-drug", "B-dos_val", "B-dos_uf", "B-rhythm_tdte"],
        extra_columns={"text": text, "label": lambda i: i % len(intents)},
        extra_features={"text": Value("string"), "label": ClassLabel(names=intents)},
    )


def dia_med():
    chapters = [
        "A00-B99  Certain infectious and parasitic diseases",
        "C00-D49  Neoplasms",
        "G00-G99  Diseases of the nervous system",
        "J00-J99  Diseases of the respiratory system",
    ]
    return build(
        {"identifier": str, "clinical_case": text, "icd-10": lambda i: i % len(chapters)},
        {"identifier": Value("string"), "clinical_case": Value("string"), "icd-10": ClassLabel(names=chapters)},
    )


def build_all():
    """Return every dataset used by the recipes, as {recipe: {subset: DatasetDict}}.

    The subset `None` stands for the datasets that have a single configuration: the
    recipes read them from `local_hf_None/`.
    """
    return {
        "cas": cas_like(["O", "B_scope_neg", "I_scope_neg"], ["O", "B_xcope_inc", "I_xcope_inc"]),
        "essai": cas_like(
            ["O", "B_cue_neg", "B_scope_neg", "I_scope_neg"],
            ["O", "B_cue_spec", "B_scope_spec", "I_scope_spec"],
        ),
        "quaero": {subset: token_classification(UMLS_TAGS) for subset in ("emea", "medline")},
        "e3c": {
            "French_clinical": token_classification(["O", "B-CLINENTITY", "I-CLINENTITY"]),
            "French_temporal": token_classification([
                "O", "B-EVENT", "B-ACTOR", "B-BODYPART", "B-TIMEX3", "B-RML",
                "I-EVENT", "I-ACTOR", "I-BODYPART", "I-TIMEX3", "I-RML",
            ]),
        },
        "mantragsc": {
            subset: token_classification(UMLS_TAGS) for subset in ("fr_emea", "fr_medline", "fr_patents")
        },
        "deft2019": {
            None: token_classification([
                "O", "B-age", "I-age", "B-genre", "I-genre", "B-issue", "I-issue", "B-origine", "I-origine",
            ]),
        },
        "deft2021": {
            "ner": token_classification(["O", "B-anatomie", "I-anatomie", "B-sosy", "I-sosy", "B-dose", "I-dose"]),
            "cls": multi_label(["immunitaire", "infections", "tumeur", "nerveux"], text_column="text"),
        },
        "morfitt": {
            "source": multi_label(["microbiology", "etiology", "virology", "physiology"], text_column="abstract"),
        },
        "pxcorpus": {None: px_corpus()},
        "diamed": {None: dia_med()},
        "frenchmedmcqa": {None: french_med_mcqa()},
        "clister": {
            "source": build(
                {"id": str, "text_1": text, "text_2": lambda i: text(i + 3), "label": lambda i: float(i % 6)},
                {"id": Value("string"), "text_1": Value("string"), "text_2": Value("string"), "label": Value("float32")},
            ),
        },
        "deft2020": {
            "task_1": build(
                {"id": str, "source": text, "cible": lambda i: text(i + 3), "moy": lambda i: float(i % 6)},
                {"id": Value("string"), "source": Value("string"), "cible": Value("string"), "moy": Value("float32")},
            ),
            "task_2": build(
                {
                    "id": str,
                    "source": text,
                    "cible_1": lambda i: text(i + 1),
                    "cible_2": lambda i: text(i + 2),
                    "cible_3": lambda i: text(i + 3),
                    "correct_cible": lambda i: i % 3,
                },
                {
                    "id": Value("string"),
                    "source": Value("string"),
                    "cible_1": Value("string"),
                    "cible_2": Value("string"),
                    "cible_3": Value("string"),
                    "correct_cible": ClassLabel(names=["1", "2", "3"]),
                },
            ),
        },
    }


@functools.lru_cache(maxsize=None)
def dataset_of(recipe, subset):
    """The dataset that a recipe reads for a subset: `None`, also as a text, for a dataset that has a single configuration."""
    return build_all()[recipe][None if subset in (None, "None") else subset]


def save_all(root):
    """Write every dataset where the recipes expect it in offline mode.

    The dataset of a recipe is saved in `<root>/<recipe>/local_hf_<subset>/`, so the
    recipe has to be run with `--data_dir <root>/<recipe>`.
    """
    for recipe, subsets in build_all().items():
        for subset, dataset in subsets.items():
            dataset.save_to_disk(str(root / recipe / f"local_hf_{subset}"))
