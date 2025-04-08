import os
import yaml
import logging
import argparse

import torch
from transformers import TrainingArguments


class TrainingArgumentsWithMPSSupport(TrainingArguments):

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="Default YAML configuration file")
    parser.add_argument("--model_name", type=str, required=False,
                        help="HuggingFace Hub model name")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Path were the model will be saved")
    parser.add_argument("--data_dir", type=str, required=False,
                        help="Path where the data are stored")
    parser.add_argument("--epochs", type=int, required=False,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, required=False,
                        help="Training batch size")
    parser.add_argument("--max_position_embeddings", type=int, required=False,
                        help="Max position embeddings")
    parser.add_argument("--weight_decay", type=float, required=False,
                        help="Weight decay")
    parser.add_argument("--learning_rate", type=float, required=False,
                        help="Learning rate")
    parser.add_argument("--subset", type=str, required=False,
                        help="Corpus subset")
    parser.add_argument("--fewshot", type=float, required=False,
                        help="Percentage of the train subset used during training")
    parser.add_argument("--offline", type=bool, required=False,
                        help="Use local huggingface dataset")
    parser.add_argument("--max_train_samples", type=int, required=False,
                        help="For debugging purposes or quicker training, truncate the number of train examples to this value if set.")
    parser.add_argument("--max_val_samples", type=int, required=False,
                        help="For debugging purposes or quicker training, truncate the number of validation examples to this value if set.")
    parser.add_argument("--max_test_samples", type=int, required=False,
                        help="For debugging purposes or quicker evaluation, truncate the number of test examples to this value if set.")

    # Load command line arguments
    args_cli = parser.parse_args()
    args_cli = vars(args_cli)

    args = {}
    # Load local config arguments is any
    args_local = {}
    if args_cli["config"]:
        args_local = yaml.safe_load(open(args_cli["config"]))
    # Load global config arguments
    args_global = yaml.safe_load(open("../../../config.yaml"))

    # Update args with local arguments (which contains great default for each task)
    args.update({k: v for k, v in args_local.items() if v is not None})
    # Overwrite local args with global arguments (which allow changing params for all tasks)
    args.update({k: v for k, v in args_global.items() if v is not None})
    # Overwrite local and global with command line arguments (which allow precise param changing)
    #  and fills missing arguments with default values
    args.update({k: v for k, v in args_cli.items() if v is not None or k not in args})

    if args['max_train_samples'] is not None and (args['fewshot'] is not None and args['fewshot'] != 1.0):
        raise ValueError('Cannot use `max_train_samples` and `fewshot` at the same time. Please check local and global config files.')

    args["output_dir"] = args["output_dir"].rstrip('/')

    # Resolve path to model, it can be either hub, full path, rel path
    m = args["model_name"]
    # If path exists, no problem
    if not os.path.exists(m):
        # If path not exist, it's either a model from the hub model or a
        #  relative path that broke since we `cd` to `recipes/task/scripts`
        # Try to fix the relative path
        fixed_path = os.path.join('..', '..', '..', m)
        if os.path.exists(fixed_path):
            # Great it's a local path
            m = fixed_path
        else:
            # Still do not exist, it must me a model from the hub
            if args['offline']:
                # If online
                model_name_clean = m.lower().replace('/', '_')
                m = os.path.join('..', '..', '..', 'models', model_name_clean)
    args['model_name'] = m

    if args["offline"]:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # transformers<4.42
        os.environ['HF_HUB_OFFLINE'] = '1'  # datasets>=2.21, transformers>=4.42

    # print(f">> Model path: >>{args['model_name']}<<")

    return argparse.Namespace(**args)
