import logging

from transformers import AutoTokenizer, AutoModel


def save_locally(model_name):
    logging.info(f">> Downloading {model_name}")
    local_path = f"./models/{model_name.lower().replace('/','_')}/"

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

    with open('models.txt') as f_in:
        models = [l.strip() for l in f_in if l.strip()]

    for m in models:
        save_locally(m)
