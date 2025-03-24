import os
import logging

try:
    with open('models.txt') as f:
        MODELS_TO_EVALUATE = [l.strip() for l in f if not l.startswith('#') and l.strip()]
except FileNotFoundError:
    # logging.warning('`models.txt` not found. Please provide models in --models')
    MODELS_TO_EVALUATE = []

task2script = {
    'CAS_POS': './run_task_1.sh {model_name}',
    'CAS_CLS': './run_task_2.sh {model_name}',
    'CAS_NER NEG': './run_task_3.sh {model_name}',
    'CAS_NER SPEC': './run_task_4.sh {model_name}',

    'CLISTER': './run.sh {model_name}',

    'Diamed': './run.sh {model_name}',

    'E3C_French_clinical': './run.sh {model_name} French_clinical',
    'E3C_French_temporal': './run.sh {model_name} French_temporal',

    'ESSAI_POS': './run_task_1.sh {model_name}',
    'ESSAI_CLS': './run_task_2.sh {model_name}',
    'ESSAI_NER NEG': './run_task_3.sh {model_name}',
    'ESSAI_NER SPEC': './run_task_4.sh {model_name}',

    'FrenchMedMCQA_MCQA': './run_task_1.sh {model_name}',
    'FrenchMedMCQA_CLS': './run_task_2.sh {model_name}',

    'MantraGSC_fr_emea': './run.sh {model_name} fr_emea',
    'MantraGSC_fr_medline': './run.sh {model_name} fr_medline',
    'MantraGSC_fr_patents': './run.sh {model_name} fr_patents',

    'Morfitt': './run.sh {model_name}',

    'PXCorpus_NER': './run_task_1.sh {model_name}',
    'PXCorpus_CLS': './run_task_2.sh {model_name}',

    'QUAERO_EMEA': './run.sh {model_name} emea',
    'QUAERO_MEDLINE': './run.sh {model_name} medline',
}


if __name__ == '__main__':
    import argparse

    def arguments():
        parser = argparse.ArgumentParser(
            description='Run selected DrBenchmark\'s task on selected models.')
        parser.add_argument("--tasks", type=str, nargs='+', required=False, help="Tasks to evaluate or 'all'", choices=list(task2script.keys()) + ['all'])
        parser.add_argument("--models", type=str, nargs='+', required=False, help="Space separated list of models to evaluate (overrides models defined in ./models.txt)", default=MODELS_TO_EVALUATE)
        parser.add_argument("--nb-run", type=int, required=False, help="Number of run to launch", default=1)
        parser.add_argument("--log-level", type=str, required=False, help="Level of logging (default: INFO)", default='INFO')
        args = parser.parse_args()

        return args

    args = arguments()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s::%(message)s',
        level=logging.getLevelName(args.log_level),
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.tasks == 'all':
        args.tasks = list(task2script.keys())

    for task in args.tasks:
        corpus, subset = (task if '_' in task else task + '_').split('_', 1)

        for model in args.models:
            for iteration in range(args.nb_run):
                logging.info('=============================')
                logging.info(f'Evaluating {model} on {corpus}-{subset} (run {iteration})')
                logging.info('=============================')
                logging.info('')

                os.system(f'cd recipes/{corpus.lower()}/scripts/ ; {task2script[task].format(model_name=model)}')
