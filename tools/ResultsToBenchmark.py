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


if __name__ == '__main__':
    print("Loading results...")
    with open('stats/results.json') as f:
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

    """
    #Only keep 4 runs and compute score's mean
    from run import task2script
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].apply(len)
    nb_runs = nb_runs[nb_runs < 4].reset_index().drop(columns='metric').drop_duplicates()
    for i, r in nb_runs.iterrows():
        corpus = r['dataset']
        task = r['task']
        model = r['model']
        comm = f'cd recipes/{corpus}/scripts/ ; ' + task2script[corpus+'-'+task].format(model_name=model)
        print((comm + '\n') * r['score'])
    """

    print("Filtering out experiments with less than 4 runs...")
    nb_runs = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].transform(len)
    print(
        df.assign(n=nb_runs)[nb_runs < 4][['model', 'dataset', 'task', 'n']].drop_duplicates()
    )

    df = df[nb_runs >= 4]

    print("Randomly choosing only 4 runs if more are available...")
    df = df.groupby(['model', 'dataset', 'task', 'fewshot', 'metric'])['score'].apply(lambda x: x.sample(n=4).mean())

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
    print("Dumping to stats/overall_results.xlsx...")
    df.to_excel('stats/overall_results.xlsx')
