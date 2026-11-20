import collections
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile

# Root of the repository: `models.txt` and the recipes are found there, wherever `run.py` is run from
REPO_DIR = os.path.dirname(os.path.realpath(__file__))

# What `run.py` starts for a task: the script of its recipe, with its YAML file and, for a script that
# reads several subsets of its dataset, the subset
Script = collections.namedtuple('Script', 'name config subset', defaults=[None])

# The tasks. `<recipe>-<task>`: the script is in recipes/<recipe>/scripts/ and the YAML file in recipes/<recipe>/yaml/
task2script = {
    'cas-pos': Script('task_1_finetuning_bert_pos.py', 'pos.yaml'),
    'cas-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),
    'cas-ner_neg': Script('task_3_finetuning_bert_ner.py', 'ner_neg.yaml'),
    'cas-ner_spec': Script('task_3_finetuning_bert_ner.py', 'ner_spec.yaml'),

    'clister-regr': Script('finetuning_bert_regr.py', 'regr.yaml'),

    'diamed-cls': Script('task_1_finetuning_bert_cls.py', 'cls.yaml'),

    'e3c-ner_clinical': Script('finetuning_bert_ner.py', 'ner.yaml', 'French_clinical'),
    'e3c-ner_temporal': Script('finetuning_bert_ner.py', 'ner.yaml', 'French_temporal'),

    'essai-pos': Script('task_1_finetuning_bert_pos.py', 'pos.yaml'),
    'essai-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),
    'essai-ner_neg': Script('task_3_finetuning_bert_ner.py', 'ner_neg.yaml'),
    'essai-ner_spec': Script('task_3_finetuning_bert_ner.py', 'ner_spec.yaml'),

    'frenchmedmcqa-mcqa': Script('task_1_finetuning_bert_mcqa.py', 'mcqa.yaml'),
    'frenchmedmcqa-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),

    'mantragsc-ner_emea': Script('finetuning_bert_ner.py', 'ner.yaml', 'fr_emea'),
    'mantragsc-ner_medline': Script('finetuning_bert_ner.py', 'ner.yaml', 'fr_medline'),
    'mantragsc-ner_patents': Script('finetuning_bert_ner.py', 'ner.yaml', 'fr_patents'),

    'morfitt-cls': Script('finetuning_bert_cls.py', 'cls.yaml'),

    'deft2019-ner': Script('finetuning_bert_ner.py', 'ner.yaml'),

    'deft2020-regr': Script('task_1_finetuning_bert_regr.py', 'regr.yaml'),
    'deft2020-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),

    'deft2021-ner': Script('task_1_finetuning_bert_ner.py', 'ner.yaml'),
    'deft2021-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),

    'pxcorpus-ner': Script('task_1_finetuning_bert_ner.py', 'ner.yaml'),
    'pxcorpus-cls': Script('task_2_finetuning_bert_cls.py', 'cls.yaml'),

    'quaero-ner_emea': Script('finetuning_bert_ner.py', 'ner.yaml', 'emea'),
    'quaero-ner_medline': Script('finetuning_bert_ner.py', 'ner.yaml', 'medline'),
}

# Options of the recipes that are not options of `run.py`: it sets the first four itself for
# each run, and each recipe has its own directories
NOT_OPTIONS_OF_RUN_PY = ('config', 'model_name', 'subset', 'seed', 'output_dir', 'run_dir', 'data_dir')


def scripts_dir(task):
    """Directory of the scripts of the recipe of a task: a run is started there."""
    return os.path.join(REPO_DIR, 'recipes', task.split('-')[0].lower(), 'scripts')


def command(task, model, options=()):
    """The command of a run of a task for a model, to start in `scripts_dir(task)`: its program and its arguments.

    The script is started by the Python that runs `run.py`, without a shell: nothing else is
    needed, neither bash nor the launcher of the task. `options` are given to the script as they are.
    """
    script = task2script[task]
    arguments = [sys.executable, script.name, '--config', os.path.join('..', 'yaml', script.config), '--model_name', model]
    if script.subset is not None:
        arguments += ['--subset', script.subset]
    return arguments + list(options)


def load_recipes_utils():
    """The module `recipes/common/utils.py`, which defines the options of the recipes."""
    path = os.path.join(REPO_DIR, 'recipes', 'common', 'utils.py')
    spec = importlib.util.spec_from_file_location('drbenchmark_recipes_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    import argparse

    # The code shared by the recipes: their options, and how a run says what it did
    recipes_utils = load_recipes_utils()

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
        parser.add_argument("--job", type=int, required=False,
                            help="Only do the run that has this number, from 0, among the runs that are asked for: they "
                                 "are numbered task by task, then model by model. A job of a job array of SLURM does "
                                 "the run that has its number this way: see run_all_slurm.sh.", default=None)
        parser.add_argument("--log-level", type=str, required=False,
                            help="Level of logging (default: INFO)", default='INFO')
        recipes = parser.add_argument_group(
            'options of the recipes',
            'They are given to every run, where they take the place of the values of the YAML file of its task '
            'and of config.yaml.')
        given_to_the_recipes = recipes_utils.add_arguments(recipes, but=NOT_OPTIONS_OF_RUN_PY)
        args = parser.parse_args()

        # The seeds of the runs: `transformers.set_seed` accepts the ones below 2**32
        if args.seed < 0 or args.seed + args.nb_run > 2**32:
            parser.error(f'the seeds of the runs have to be between 0 and 2**32-1: --seed {args.seed} --nb-run {args.nb_run}')
        if args.job is not None and args.job < 0:
            parser.error(f'the runs are numbered from 0: --job {args.job}')

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

    # Before the first message: a message sets the default format and level up otherwise, and hides the next ones
    logging.basicConfig(
        format='%(asctime)s %(levelname)s::%(message)s',
        level=logging.getLevelName(args.log_level),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.models is None:
        try:
            with open(os.path.join(REPO_DIR, 'models.txt')) as f:
                # One model per line, but the empty lines and the comments
                args.models = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        except FileNotFoundError:
            logging.warning('`models.txt` not found. Please provide models in --models')
            args.models = []
    elif not os.path.samefile(os.getcwd(), REPO_DIR):
        args.models = [from_repository(model) for model in args.models]

    if args.tasks == ['all']:
        args.tasks = sorted(list(task2script.keys()))

    # The runs that are asked for, in the order they are done: (task, model, number of the run)
    runs = [(task, model, iteration) for task in args.tasks for model in args.models for iteration in range(args.nb_run)]
    if args.job is not None:
        # The last job of the job array of SLURM, when `run.py` is run by one
        last_job = os.environ.get('SLURM_ARRAY_TASK_MAX', '')
        if last_job.isdigit() and int(last_job) < len(runs) - 1:
            logging.warning(
                f'The job array stops at the job {last_job}: the runs {int(last_job) + 1} to {len(runs) - 1} have no job. '
                f'There is one job per run with --array=0-{len(runs) - 1}')
        if args.job >= len(runs):
            logging.warning(f'There is no run {args.job}: {len(runs)} runs are asked for, numbered from 0')
        runs = runs[args.job:args.job + 1]

    # Runs that did not end properly: (task, model, iteration, exit code)
    failures = []
    # Runs whose results existed: the recipe did not compute them again
    already_computed = 0

    with tempfile.TemporaryDirectory(prefix='drbenchmark_') as directory:
        # Where each run writes what it did: see `report_status` in recipes/common/utils.py
        status_file = os.path.join(directory, 'status.json')
        environ = dict(os.environ, **{recipes_utils.STATUS_FILE: status_file})

        for task, model, iteration in runs:
            corpus, subset = task.split('-')
            # Each run has its own seed, always the same: the recipe names its results after its
            # arguments, and does not compute again a run whose results exist
            seed = args.seed + iteration
            logging.info('=============================')
            logging.info(f'Evaluating {model} on {corpus}-{subset} (run {iteration}, seed {seed})')
            logging.info('=============================')
            logging.info('')

            if os.path.exists(status_file):
                os.remove(status_file)
            try:
                returncode = subprocess.call(
                    command(task, model, ['--seed', str(seed), *recipe_options]), cwd=scripts_dir(task), env=environ)
            except KeyboardInterrupt:
                logging.info('KeyboardInterrupt Exiting...')
                sys.exit(130)

            if returncode != 0:
                logging.error(f'Evaluation of {model} on {corpus}-{subset} (run {iteration}) failed with exit code {returncode}')
                failures.append((task, model, iteration, returncode))
            elif os.path.exists(status_file):
                with open(status_file, encoding='utf-8') as f:
                    already_computed += json.load(f)['status'] == recipes_utils.ALREADY_COMPUTED

    computed = len(runs) - already_computed - len(failures)
    logging.info(f'{len(runs)} run(s): {computed} computed, {already_computed} already computed, {len(failures)} failed')

    # A failed run does not stop the other ones, but it must not go unnoticed
    if failures:
        logging.error(f'{len(failures)} run(s) failed:')
        for task, model, iteration, returncode in failures:
            logging.error(f'  {task} | {model} | run {iteration} | exit code {returncode}')
        sys.exit(1)
