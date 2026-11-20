"""Save models of the HuggingFace Hub in `models/`, where the recipes read them in offline mode.

The models are the ones of `models.txt`, or the ones that are given with `--models`.
"""

import os
import sys
import logging
import argparse

from transformers import AutoTokenizer, AutoModel

# Root of the repository: `models.txt` is read there and the models are saved in `models/`,
# wherever the script is run from
REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def models_to_download():
    """Models of `models.txt`: one per line, but the empty lines and the comments (`#`), as for `run.py`."""
    with open(os.path.join(REPO_DIR, 'models.txt')) as f_in:
        lines = [line.strip() for line in f_in]
    return [line for line in lines if line and not line.startswith('#')]


def save_locally(model_name):
    logging.info(f">> Downloading {model_name}")
    local_path = os.path.join(REPO_DIR, "models", model_name.lower().replace('/', '_'))

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, nargs='+', required=False, default=None, metavar="MODEL",
                        help="Names, on the HuggingFace Hub, of the models to save (default: the models of models.txt).")
    return parser.parse_args()


def main():
    args = arguments()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    models = args.models
    if models is None:
        try:
            models = models_to_download()
        except FileNotFoundError:
            sys.exit(f"`models.txt` is not in {REPO_DIR}: give the models to save with --models.")
    for m in models:
        save_locally(m)


if __name__ == '__main__':
    main()
