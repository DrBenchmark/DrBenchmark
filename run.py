import importlib.util
import logging
import os
import shlex
import subprocess
import sys

# Root of the repository: `models.txt` and the recipes are found there, wherever `run.py` is run from
REPO_DIR = os.path.dirname(os.path.realpath(__file__))

task2script = {
    'cas-pos': './run_task_1.sh {model_name}',
    'cas-cls': './run_task_2.sh {model_name}',
    'cas-ner_neg': './run_task_3.sh {model_name}',
    'cas-ner_spec': './run_task_4.sh {model_name}',

    'clister-regr': './run.sh {model_name}',

    'diamed-cls': './run.sh {model_name}',

    'e3c-ner_clinical': './run.sh {model_name} French_clinical',
    'e3c-ner_temporal': './run.sh {model_name} French_temporal',

    'essai-pos': './run_task_1.sh {model_name}',
    'essai-cls': './run_task_2.sh {model_name}',
    'essai-ner_neg': './run_task_3.sh {model_name}',
    'essai-ner_spec': './run_task_4.sh {model_name}',

    'frenchmedmcqa-mcqa': './run_task_1.sh {model_name}',
    'frenchmedmcqa-cls': './run_task_2.sh {model_name}',

    'mantragsc-ner_emea': './run.sh {model_name} fr_emea',
    'mantragsc-ner_medline': './run.sh {model_name} fr_medline',
    'mantragsc-ner_patents': './run.sh {model_name} fr_patents',

    'morfitt-cls': './run.sh {model_name}',

    'deft2019-ner': './run.sh {model_name}',

    'deft2020-regr': './run_task_1.sh {model_name}',
    'deft2020-cls': './run_task_2.sh {model_name}',

    'deft2021-ner': './run_task_1.sh {model_name}',
    'deft2021-cls': './run_task_2.sh {model_name}',

    'pxcorpus-ner': './run_task_1.sh {model_name}',
    'pxcorpus-cls': './run_task_2.sh {model_name}',

    'quaero-ner_emea': './run.sh {model_name} emea',
    'quaero-ner_medline': './run.sh {model_name} medline',
}

# Options of the recipes that are not options of `run.py`: it sets the first four itself for
# each run, and each recipe has its own directories
NOT_OPTIONS_OF_RUN_PY = ('config', 'model_name', 'subset', 'seed', 'output_dir', 'run_dir', 'data_dir')


def load_recipes_utils():
    """The module `recipes/common/utils.py`, which defines the options of the recipes."""
    path = os.path.join(REPO_DIR, 'recipes', 'common', 'utils.py')
    spec = importlib.util.spec_from_file_location('drbenchmark_recipes_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    import argparse

    try:
        with open(os.path.join(REPO_DIR, 'models.txt')) as f:
            # One model per line, but the empty lines and the comments
            MODELS_TO_EVALUATE = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    except FileNotFoundError:
        logging.warning('`models.txt` not found. Please provide models in --models')
        MODELS_TO_EVALUATE = []

    def arguments():
        """The options of `run.py`, and the names of the ones that are given to the recipes."""
        parser = argparse.ArgumentParser(
            description='Run selected DrBenchmark\'s task on selected models.')
        parser.add_argument("--tasks", type=str, nargs='+', required=True, metavar="TASK",
                            help="Tasks to evaluate or 'all'. Choose from %(choices)s", choices=list(task2script.keys()) + ['all'])
        parser.add_argument("--models", type=str, nargs='+', required=False,
                            help="Name or path of models to evaluate (overrides models defined in models.txt)", default=None)
        parser.add_argument("--nb-run", type=int, required=False,
                            help="Number of runs of each task for each model (default: 1). A run which was already "
                                 "computed with its seed is not computed again.", default=1)
        parser.add_argument("--seed", type=int, required=False,
                            help="Seed of the first run of each task for each model: the next runs have the next "
                                 "seeds (default: 0)", default=0)
        parser.add_argument("--log-level", type=str, required=False,
                            help="Level of logging (default: INFO)", default='INFO')
        recipes = parser.add_argument_group(
            'options of the recipes',
            'They are given to every run, where they take the place of the values of the YAML file of its task '
            'and of config.yaml.')
        given_to_the_recipes = load_recipes_utils().add_arguments(recipes, but=NOT_OPTIONS_OF_RUN_PY)
        args = parser.parse_args()

        # The seeds of the runs: `transformers.set_seed` accepts the ones below 2**32
        if args.seed < 0 or args.seed + args.nb_run > 2**32:
            parser.error(f'the seeds of the runs have to be between 0 and 2**32-1: --seed {args.seed} --nb-run {args.nb_run}')

        return args, given_to_the_recipes

    def from_repository(model):
        """A model given by a path relative to the current directory, written from the root of the repository.

        The recipes do not run in the current directory: they look for a relative path from
        the root of the repository. A name of the HuggingFace Hub is not a path that exists.
        """
        if os.path.isabs(model) or not os.path.exists(model):
            return model
        relative = os.path.relpath(model, REPO_DIR)
        return relative + '/' if model.endswith('/') else relative

    args, given_to_the_recipes = arguments()
    # The options that were typed, as the recipes read them: `--epochs 2`
    recipe_options = [
        argument
        for name in given_to_the_recipes if getattr(args, name) is not None
        for argument in (f'--{name}', str(getattr(args, name)))
    ]

    if args.models is None:
        args.models = MODELS_TO_EVALUATE
    elif not os.path.samefile(os.getcwd(), REPO_DIR):
        args.models = [from_repository(model) for model in args.models]

    logging.basicConfig(
        format='%(asctime)s %(levelname)s::%(message)s',
        level=logging.getLevelName(args.log_level),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.tasks == ['all']:
        args.tasks = sorted(list(task2script.keys()))

    # Runs that did not end properly: (task, model, iteration, exit code)
    failures = []

    for task in args.tasks:
        corpus, subset = task.split('-')

        for model in args.models:
            for iteration in range(args.nb_run):
                # Each run has its own seed, always the same: the recipe names its results after its
                # arguments, and does not compute again a run whose results exist
                seed = args.seed + iteration
                logging.info('=============================')
                logging.info(f'Evaluating {model} on {corpus}-{subset} (run {iteration}, seed {seed})')
                logging.info('=============================')
                logging.info('')

                command = task2script[task].format(model_name=shlex.quote(model))
                # What follows the model and the subset is given to the script of the recipe
                command += ' ' + shlex.join(['--seed', str(seed), *recipe_options])
                scripts_dir = os.path.join(REPO_DIR, 'recipes', corpus.lower(), 'scripts')
                try:
                    returncode = subprocess.call(command, shell=True, cwd=scripts_dir)
                except KeyboardInterrupt:
                    logging.info('KeyboardInterrupt Exiting...')
                    sys.exit(130)

                if returncode != 0:
                    logging.error(f'Evaluation of {model} on {corpus}-{subset} (run {iteration}) failed with exit code {returncode}')
                    failures.append((task, model, iteration, returncode))

    # A failed run does not stop the other ones, but it must not go unnoticed
    if failures:
        logging.error(f'{len(failures)} run(s) failed:')
        for task, model, iteration, returncode in failures:
            logging.error(f'  {task} | {model} | run {iteration} | exit code {returncode}')
        sys.exit(1)
