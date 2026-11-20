import os
import sys
import yaml
import json
import random
import hashlib
import argparse


# Root of the repository: this file is `recipes/common/utils.py`
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

subset2task_detail = {
    'emea': 'emea', 'medline': 'medline',  # pxcorpus
    'fr_emea': 'emea', 'fr_medline': 'medline', 'fr_patents': 'patents',  # mantragsc
    'French_clinical': 'clinical', 'French_temporal': 'temporal'  # e3c
}


# Arguments that do not take part in the name of a run: they describe it, without changing what it computes
NOT_IN_RUN_NAME = ("hub_model_name",)

# `run.py` gives, in this variable of the environment, the path of a file where the run it starts
# writes what it did: it counts this way the runs that it did not have to compute again
STATUS_FILE = "DRBENCHMARK_RUN_STATUS"
ALREADY_COMPUTED, COMPUTED = "already computed", "computed"


def report_status(status, run_path):
    """Write what the run did in the file that `run.py` asks for, when it is the one that started the run."""
    path = os.environ.get(STATUS_FILE)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"status": status, "results": os.path.abspath(run_path)}, f)


def create_run_name(args):
    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    # Creating the hash using the parameters (including the seed, dataset, model, ...)
    hashed = {name: value for name, value in args.items() if name not in NOT_IN_RUN_NAME}
    hash_ = hashlib.shake_128(
        json.dumps(hashed, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest(8)
    output_name = f'DrBenchmark-{args["task"]}-{hash_}'
    return output_name


def get_run_path(args):
    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    output_name = create_run_name(args)
    return os.path.join(args['run_dir'], f'{output_name}.json')


def get_model_path(args):
    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    output_name = create_run_name(args)
    return os.path.join(args['output_dir'], f'{output_name}')


def str2bool(value):
    """Parse a boolean given on the command line, e.g. `--offline False`.

    `type=bool` cannot be used for that: it turns any non-empty string, even "False", into True.
    """
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got '{value}'")


def relative_path(path, start):
    """`path` written from the directory `start`, with its final slash if it has one."""
    relative = os.path.relpath(path, start)
    return relative + '/' if path.endswith('/') else relative


def is_in_repository(path):
    return os.path.commonpath([os.path.realpath(path), REPO_DIR]) == REPO_DIR


def move_to_scripts_dir(args_cli):
    """Make the `scripts/` directory of the running recipe the current directory.

    A recipe writes its paths from its `scripts/` directory: in its YAML file (`../runs/`),
    and in its results (`../../../models/<model>`). They are kept as they are, so that
    neither the name of a run, which is built from its arguments, nor the content of its
    results depend on the directory the script is started from. The paths given on the
    command line are relative to that directory: they are rewritten from `scripts/`.
    """
    scripts_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    if os.path.dirname(os.path.dirname(scripts_dir)) != os.path.join(REPO_DIR, 'recipes'):
        # Not started as the script of a recipe (`recipes/<recipe>/scripts/<script>.py`):
        # the paths are left as they are, relative to the current directory
        return
    started_from_scripts_dir = os.path.samefile(os.getcwd(), scripts_dir)

    # The YAML file of a recipe is always written from `scripts/` (`../yaml/pos.yaml`),
    # whatever the way the launcher or the user spells its path
    config = args_cli["config"]
    if is_in_repository(config):
        args_cli["config"] = relative_path(os.path.realpath(config), scripts_dir)
    elif not os.path.isabs(config) and not started_from_scripts_dir:
        args_cli["config"] = relative_path(config, scripts_dir)

    if started_from_scripts_dir:
        return

    for name in ("output_dir", "run_dir", "data_dir"):
        path = args_cli[name]
        if path is not None and not os.path.isabs(path):
            args_cli[name] = relative_path(path, scripts_dir)
    # A model given by a relative path. A name of the HuggingFace Hub is not a path that exists.
    model = args_cli["model_name"]
    if model is not None and not os.path.isabs(model) and os.path.exists(model):
        args_cli["model_name"] = relative_path(model, scripts_dir)

    os.chdir(scripts_dir)


def add_arguments(parser, but=()):
    """Add the options of the recipes to `parser`, but the ones named in `but`, and return their names.

    `run.py` accepts the same options, which it gives to the recipes: it loads this module
    for that, which must not import what is slow to load (`torch`, `transformers`).
    """
    names = []

    def add(option, **kwargs):
        name = option[2:]
        if name not in but:
            parser.add_argument(option, **kwargs)
            names.append(name)

    add("--config", type=str, required=True,
        help="Default YAML configuration file")
    add("--model_name", type=str, required=False,
        help="HuggingFace Hub model name")
    add("--output_dir", type=str, required=False,
        help="Path were the model will be saved")
    add("--run_dir", type=str, required=False,
        help="Path were the run output will be saved")
    add("--data_dir", type=str, required=False,
        help="Path where the data are stored")
    add("--seed", type=int, required=False,
        help="Seed to be used (between 0 and 2**32-1) in `transformers.set_seed` for reproducibility.")
    add("--epochs", type=int, required=False,
        help="Training epochs")
    add("--batch_size", type=int, required=False,
        help="Training batch size")
    add("--gradient_accumulation_steps", type=int, required=False,
        help="Number of batches before update model's weights: "
             "the effective batch size is batch_size * gradient_accumulation_steps.")
    add("--max_position_embeddings", type=int, required=False,
        help="Max position embeddings")
    add("--weight_decay", type=float, required=False,
        help="Weight decay")
    add("--learning_rate", type=float, required=False,
        help="Learning rate")
    add("--subset", type=str, required=False,
        help="Corpus subset")
    add("--fewshot", type=float, required=False,
        help="Percentage of the train subset used during training")
    add("--offline", type=str2bool, required=False, metavar="{true,false}",
        help="Use the datasets and models saved locally instead of the HuggingFace Hub")
    add("--max_train_samples", type=int, required=False,
        help="For debugging purposes or quicker training, truncate the number of train examples to this value if set.")
    add("--max_val_samples", type=int, required=False,
        help="For debugging purposes or quicker training, truncate the number of validation examples to this value if set.")
    add("--max_test_samples", type=int, required=False,
        help="For debugging purposes or quicker evaluation, truncate the number of test examples to this value if set.")

    return names


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)

    # Load command line arguments
    args_cli = parser.parse_args()
    args_cli = vars(args_cli)

    args = {}
    # Load local config arguments is any
    args_local = {}
    if args_cli["config"]:
        args_local = yaml.safe_load(open(args_cli["config"]))
    # Load global config arguments
    args_global = yaml.safe_load(open(os.path.join(REPO_DIR, "config.yaml")))

    # From here, the current directory is the `scripts/` directory of the recipe
    move_to_scripts_dir(args_cli)

    # Update args with local arguments (which contains great default for each task)
    args.update(args_local)
    # Overwrite local args with global arguments (which allow changing params for all tasks)
    args.update(args_global)
    # Overwrite local and global with command line arguments (which allow precise param changing)
    #  and fills missing arguments with default values
    args.update({k: v for k, v in args_cli.items() if v is not None or k not in args})

    # Without gradient accumulation, the weights are updated after each batch
    if args['gradient_accumulation_steps'] is None:
        args['gradient_accumulation_steps'] = 1
    if args['gradient_accumulation_steps'] < 1:
        raise ValueError(f"`gradient_accumulation_steps` must be at least 1, got {args['gradient_accumulation_steps']}.")

    if args['max_train_samples'] is not None and (args['fewshot'] is not None and args['fewshot'] != 1.0):
        raise ValueError('Cannot use `max_train_samples` and `fewshot` at the same time. Please check local and global config files.')

    args["output_dir"] = args["output_dir"].rstrip('/')

    # Resolve path to model, it can be either hub, full path, rel path
    m = args["model_name"]
    # Name of the model on the HuggingFace Hub, when it is not given by a path
    hub_model_name = None
    # If path exists, no problem
    if not os.path.exists(m):
        # If path not exist, it's either a model from the hub model or a
        #  path relative to the root of the repository, as `run.py` gives it,
        #  while the current directory is `recipes/task/scripts`
        # Try to fix the relative path
        fixed_path = os.path.join('..', '..', '..', m)
        if os.path.exists(fixed_path):
            # Great it's a local path
            m = fixed_path
        else:
            # Still do not exist, it must me a model from the hub
            hub_model_name = m
            if args['offline']:
                # If online
                model_name_clean = m.lower().replace('/', '_')
                m = os.path.join('..', '..', '..', 'models', model_name_clean)
    args['model_name'] = m
    # Written in the results of the run, for the tools to know which model it is: in offline
    # mode, `model_name` is the directory of its local copy, whose name is in lower case
    args['hub_model_name'] = hub_model_name

    if args['subset'] in subset2task_detail:
        args['task'] += '_' + subset2task_detail[args['subset']]

    if args["offline"]:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # transformers<4.42
        os.environ['HF_HUB_OFFLINE'] = '1'  # datasets>=2.21, transformers>=4.42

    if args['seed'] is None:
        # If no seed was provided: Pick a seed
        args['seed'] = random.randrange(2**32)
        # Check if it already was computed
        while os.path.exists(get_run_path(args)):
            # If yes pick another seed
            args['seed'] = (args['seed'] + 1) % (2**32)
    else:
        # If a seed was provided: checl whether it was already computed
        if os.path.exists(get_run_path(args)):
            # If yes do nothing
            print(f"This configuration was already computed (seed: {args['seed']}, model: {args['model_name']}).")
            print(json.dumps(args))
            print("Exiting...")
            report_status(ALREADY_COMPUTED, get_run_path(args))
            exit(0)
    # print(f">> Model path: >>{args['model_name']}<<")

    return argparse.Namespace(**args)
