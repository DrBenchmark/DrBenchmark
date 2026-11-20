import os
import json
import pandas as pd

from number_of_runs import NB_RUNS
from repository_paths import STATS_DIR


dataset2pretty = {
    "cas": "CAS",
    "clister": "CLISTER",
    "deft2019": "DEFT-2019",
    "deft2020": "DEFT-2020",
    "deft2021": "DEFT-2021",
    "diamed": "DiaMED",
    "e3c": "E3C",
    "essai": "ESSAI",
    "frenchmedmcqa": "FrenchMedMCQA",
    "mantragsc": "MantraGSC",
    "morfitt": "MorFITT",
    "pxcorpus": "PxCorpus",
    "quaero": "QUAERO"
}

task2pretty = {
    "cls": "CLS",
    "mcls": "Multi-Label CLS",
    "mcqa": "MCQA",
    "pos": "POS",
    "regr": "STS",

    "ner": "NER",
    "ner_clinical": "NER Clinical",
    "ner_emea": "NER EMEA",
    "ner_patents": "NER Patents",
    "ner_medline": "NER MEDLINE",
    "ner_temporal": "NER Temporal",
    "ner_neg": "NER Neg",
    "ner_spec": "NER Spec",
}

metric2pretty = {
    'edrm': 'EDRM',
    'exact_match': 'EMR',
    'hamming_score': 'Hammings',
    'overall_f1': 'F1',
    'weighted_f1': 'wF1',
    'spearman_correlation_coef': 'Spearman',
}

task2metric = {
    'cls': ('weighted_f1',),
    'mcls': ('weighted_f1',),
    'mcqa': ('hamming_score', 'exact_match'),
    'pos': ('overall_f1',),
    'regr': ('edrm', 'spearman_correlation_coef'),

    'ner': ('overall_f1',),
    'ner_clinical': ('overall_f1',),
    'ner_emea': ('overall_f1',),
    'ner_patents': ('overall_f1',),
    'ner_medline': ('overall_f1',),
    'ner_temporal': ('overall_f1',),
    'ner_neg': ('overall_f1',),
    'ner_spec': ('overall_f1',),
}

order = [
    ('cas', 'pos', 'overall_f1'),
    ('essai', 'pos', 'overall_f1'),
    ('quaero', 'ner_emea', 'overall_f1'),
    ('quaero', 'ner_medline', 'overall_f1'),
    ('e3c', 'ner_clinical', 'overall_f1'),
    ('e3c', 'ner_temporal', 'overall_f1'),
    ('morfitt', 'mcls', 'weighted_f1'),
    ('frenchmedmcqa', 'mcqa', 'hamming_score'),
    ('frenchmedmcqa', 'mcqa', 'exact_match'),
    ('frenchmedmcqa', 'cls', 'weighted_f1'),
    ('mantragsc', 'ner_emea', 'overall_f1'),
    ('mantragsc', 'ner_medline', 'overall_f1'),
    ('mantragsc', 'ner_patents', 'overall_f1'),
    ('clister', 'regr', 'edrm'),
    ('clister', 'regr', 'spearman_correlation_coef'),
    ('deft2020', 'regr', 'edrm'),
    ('deft2020', 'regr', 'spearman_correlation_coef'),
    ('deft2020', 'cls', 'weighted_f1'),
    ('deft2021', 'mcls', 'weighted_f1'),
    ('deft2021', 'ner', 'overall_f1'),
    ('diamed', 'cls', 'weighted_f1'),
    ('pxcorpus', 'ner', 'overall_f1'),
    ('pxcorpus', 'cls', 'weighted_f1'),

    # not displayed on the drbenchmark website
    ('cas', 'cls', 'weighted_f1'),
    ('cas', 'ner_neg', 'overall_f1'),
    ('cas', 'ner_spec', 'overall_f1'),
    ('essai', 'cls', 'weighted_f1'),
    ('essai', 'ner_neg', 'overall_f1'),
    ('essai', 'ner_spec', 'overall_f1')
]


if __name__ == '__main__':
    import logging

    def arguments():
        import argparse
        parser = argparse.ArgumentParser(
            description='Outputs a benchmark to "stats/overall_results_<n>runs.xlsx": the average of the runs of each '
                        'task for each model and, in a second sheet, their standard deviation.')
        parser.add_argument(
            "--results", type=str, required=False, default=os.path.join(STATS_DIR, "results.json"),
            help="Path to results.json (default: stats/results.json).")
        parser.add_argument(
            "--nb-run", type=int, required=False, default=NB_RUNS,
            help=f"Number of runs to use (default: {NB_RUNS}). Methods with less than requested runs won't be shown.")
        parser.add_argument(
            "--csv", action="store_true",
            help="Write the benchmark in csv to stdout")
        parser.add_argument(
            "--std", action="store_true",
            help="With --csv, write the standard deviation of the runs in a column after the average of each model.")
        args = parser.parse_args()

        return args

    args = arguments()
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading results...")
    with open(args.results) as f:
        res = json.load(f)

    df = pd.DataFrame(
        [[model, *task.split('|'), metric, score * 100]
            for model, tasks in res.items()
            for task, metrics in tasks.items()
            for metric, scores in metrics.items()
            for score in scores],
        columns='model dataset task fewshot metric score'.split()
    )

    # Replace cls by mcls (multi-label cls)
    df.loc[(df['dataset'] == "deft2021") & (df['task'] == "cls"), 'task'] = 'mcls'
    df.loc[(df['dataset'] == "morfitt") & (df['task'] == "cls"), 'task'] = 'mcls'

    logging.info(f"Filtering out experiments with less than {args.nb_run} runs...")
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].transform(len)
    tmp = df.assign(n=nb_runs)[nb_runs < args.nb_run][['model', 'dataset', 'task', 'n']].drop_duplicates()
    for i, sdf in tmp.groupby('model'):
        logging.info(f'For model {i}')
        logging.info(sdf[['dataset', 'task', 'n']])

    df = df[nb_runs >= args.nb_run]

    logging.info(f"Randomly choosing only {args.nb_run} runs if more are available...")
    scores = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score']
    # The runs that are chosen are the same ones for the average and for the standard deviation,
    # which is the one of a sample (n - 1): there is none for one run
    averages = scores.apply(lambda x: x.sample(n=args.nb_run, random_state=42).mean())
    deviations = scores.apply(lambda x: x.sample(n=args.nb_run, random_state=42).std())

    def table(df):
        """One row per score of the benchmark, and one column per model."""
        # Pretty table
        # Reorder lines and select metrics
        df = df.unstack([0, 3])
        df = df.reindex(index=order)
        df = df.reset_index()

        # Remove few-shot colums
        df.columns = df.columns.droplevel(1)
        df = df.round(2)
        df['task'] = df['task'].map(task2pretty.get)
        df['dataset'] = df['dataset'].map(dataset2pretty.get)
        df['metric'] = df['metric'].map(metric2pretty.get)
        return df.set_index(['dataset', 'task', 'metric'])

    df = table(averages)
    std = table(deviations)
    if args.csv:
        if args.std:
            # The standard deviation of a model follows its average: the columns are taken by their
            # position, as several ones have the name of a model when it has few-shot runs
            both = pd.concat([df, std.add_suffix(" (std)")], axis=1)
            df = both.iloc[:, [column for model in range(df.shape[1]) for column in (model, df.shape[1] + model)]]
        print(df.to_csv())
    else:
        output_file = os.path.join(STATS_DIR, f'overall_results_{args.nb_run}runs.xlsx')
        logging.info(f"Dumping to {output_file}...")
        with pd.ExcelWriter(output_file) as writer:
            # The first sheet is the table as it always was
            df.to_excel(writer, sheet_name="Sheet1")
            std.to_excel(writer, sheet_name="std")
