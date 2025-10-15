import json
import pandas as pd


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

custom_model_map = {
    "../../../models/almanach_camembert-bio-base": "almanach/camembert-bio-base",
    "../../../models/dr-bert_drbert-7gb": "Dr-BERT/DrBERT-7GB",
    "../../../models/dr-bert_drbert-7gb-large": "Dr-BERT/DrBERT-7GB-Large",
    "../../../models/flaubert_flaubert_base_cased": "flaubert/flaubert_base_cased",
    "../../../models/flaubert_flaubert_large_cased": "flaubert/flaubert_large_cased",
    "../../../models/answerdotai_modernbert-base": "answerdotai/modernbert-base",
    "../../../models/answerdotai_modernbert-large": "answerdotai/modernbert-large",
    "../../../models/thomas-sounack_bioclinical-modernbert-base": "thomas-sounack/bioclinical-modernbert-base",
    "../../../models/thomas-sounack_bioclinical-modernbert-large": "thomas-sounack/bioclinical-modernbert-large",
}


if __name__ == '__main__':
    import logging

    def arguments():
        import argparse
        parser = argparse.ArgumentParser(
            description='Outputs a benchmark to "stats/overall_results.xlsx".')
        parser.add_argument(
            "--results", type=str, required=False, default="stats/results.json",
            help="Path to results.json (default: stats/results.json).")
        parser.add_argument(
            "--nb-run", type=int, required=False, default=4,
            help="Number of runs to use (default: 4). Methods with less than requested runs won't be shown.")
        parser.add_argument(
            "--csv", action="store_true",
            help="Write the benchmark in csv to stdout")
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

    df['model'] = df['model'].map(lambda x: custom_model_map.get(x, x))
    logging.info(f"Filtering out experiments with less than {args.nb_run} runs...")
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].transform(len)
    tmp = df.assign(n=nb_runs)[nb_runs < args.nb_run][['model', 'dataset', 'task', 'n']].drop_duplicates()
    for i, sdf in tmp.groupby('model'):
        logging.info(f'For model {i}')
        logging.info(sdf[['dataset', 'task', 'n']])

    df = df[nb_runs >= args.nb_run]

    logging.info(f"Randomly choosing only {args.nb_run} runs if more are available...")
    df = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].apply(lambda x: x.sample(n=args.nb_run).mean())

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
    df = df.set_index(['dataset', 'task', 'metric'])
    if args.csv:
        print(df.to_csv())
    else:
        output_file = f'stats/overall_results_{args.nb_run}runs.xlsx'
        logging.info(f"Dumping to {output_file}...")
        df.to_excel(output_file)
