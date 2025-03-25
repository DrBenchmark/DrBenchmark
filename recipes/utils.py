import os
import argparse
from argparse import Namespace

import yaml
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

    parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",                  type=str,   required=True,  help="Default YAML configuration file")
    parser.add_argument("--model_name",              type=str,   required=False, help="HuggingFace Hub model name")
    parser.add_argument("--output_dir",              type=str,   required=False, help="Path were the model will be saved")
    parser.add_argument("--data_dir",                type=str,   required=False, help="Path where the data are stored")
    parser.add_argument("--epochs",                  type=int,   required=False, help="Training epochs")
    parser.add_argument("--batch_size",              type=int,   required=False, help="Training batch size")
    parser.add_argument("--max_position_embeddings", type=int,   required=False, help="Max position embeddings")
    parser.add_argument("--weight_decay",            type=float, required=False, help="Weight decay")
    parser.add_argument("--learning_rate",           type=float, required=False, help="Learning rate")
    parser.add_argument("--subset",                  type=str,   required=False, help="Corpus subset")
    parser.add_argument("--fewshot",                 type=float, required=False, help="Percentage of the train subset used during training", default=1.0)
    parser.add_argument("--offline",                 type=bool,  required=False, help="Use local huggingface dataset", default=False)

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

    for k in args.keys():

        if args[k] == None:
            args[k] = args_yaml[k]
    
    args["output_dir"] = args["output_dir"].rstrip('/')
    
    if args["offline"] == True:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ['TRANSFORMERS_OFFLINE']='1'
        model_name_clean = args['model_name'].lower().replace('/','_')
        args["model_name"] = f"../../../models/{model_name_clean}"

    print(f">> Model path: >>{args['model_name']}<<")
    
    return Namespace(**args)
