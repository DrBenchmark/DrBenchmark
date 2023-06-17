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

    args = parser.parse_args()
    args = vars(args)

    overall_args_yaml = yaml.load(open("../../../config.yaml"), Loader=yaml.FullLoader)
    args["offline"] = overall_args_yaml["offline"]

    args_yaml = yaml.load(open(args["config"]), Loader=yaml.FullLoader)

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
