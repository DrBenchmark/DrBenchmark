import logging
import subprocess

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


if __name__ == '__main__':
    import argparse

    try:
        with open('models.txt') as f:
            MODELS_TO_EVALUATE = [l.strip() for l in f if not l.startswith('#') and l.strip()]
    except FileNotFoundError:
        logging.warning('`models.txt` not found. Please provide models in --models')
        MODELS_TO_EVALUATE = []

    def arguments():
        parser = argparse.ArgumentParser(
            description='Run selected DrBenchmark\'s task on selected models.')
        parser.add_argument("--tasks", type=str, nargs='+', required=True, metavar="TASK",
                            help="Tasks to evaluate or 'all'. Choose from %(choices)s", choices=list(task2script.keys()) + ['all'])
        parser.add_argument("--models", type=str, nargs='+', required=False,
                            help="Name or path of models to evaluate (overrides models defined in ./models.txt)", default=MODELS_TO_EVALUATE)
        parser.add_argument("--nb-run", type=int, required=False,
                            help="Number of runs to launch", default=1)
        parser.add_argument("--log-level", type=str, required=False,
                            help="Level of logging (default: INFO)", default='INFO')
        args = parser.parse_args()

        return args

    args = arguments()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s::%(message)s',
        level=logging.getLevelName(args.log_level),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.tasks == ['all']:
        args.tasks = sorted(list(task2script.keys()))

    for task in args.tasks:
        corpus, subset = (task if '_' in task else task + '_').split('_', 1)

        for model in args.models:
            for iteration in range(args.nb_run):
                logging.info('=============================')
                logging.info(f'Evaluating {model} on {corpus}-{subset} (run {iteration})')
                logging.info('=============================')
                logging.info('')

                command = f'cd recipes/{corpus.lower()}/scripts/ ; {task2script[task].format(model_name=model)}'
                try:
                    subprocess.call(command, shell=True)
                except KeyboardInterrupt:
                    logging.info('KeyboardInterrupt Exiting...')
                    exit()
