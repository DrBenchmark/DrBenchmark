import os
import logging

from transformers import AutoTokenizer, AutoModel

# Root of the repository: `models.txt` is read there and the models are saved in `models/`,
# wherever the script is run from
REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def save_locally(model_name):
    logging.info(f">> Downloading {model_name}")
    local_path = os.path.join(REPO_DIR, "models", model_name.lower().replace('/', '_'))

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(local_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_path)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    with open(os.path.join(REPO_DIR, 'models.txt')) as f_in:
        models = [l.strip() for l in f_in if l.strip()]

    for m in models:
        save_locally(m)
