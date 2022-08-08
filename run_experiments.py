import os
import sys
from itertools import product
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from data_formatting import LABEL_COL
from data_preprocessor import build_data_preprocessor
from experiments_settings import DATASETS_FILES, FEATURES_SELECTORS, MODELS, KS, get_cv, get_scoring_metrics, N_JOBS


def run_all(logs_dir='logs', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    if len(sys.argv) == 1:
        datasets_files = DATASETS_FILES
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]

    for experiment_args in product(MODELS.keys(), datasets_files, FEATURES_SELECTORS.keys(), KS):
        print(f'Start Experiment, Settings: {experiment_args}')
        output_log_file = run_experiment(*experiment_args, logs_dir=logs_dir, overwrite_logs=overwrite_logs)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(estimator_name, filename, fs_name, k, logs_dir=None, overwrite_logs=True):
    log_filename = f'{"_".join([estimator_name, Path(filename).name, fs_name, str(k)])}.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    df = pd.read_csv(filename)
    cv = get_cv(df)
    X = df.drop(columns=[LABEL_COL])
    y = pd.Series(LabelEncoder().fit_transform(df[LABEL_COL]))

    cachedir1, cachedir2 = mkdtemp(), mkdtemp()
    pipeline = Pipeline(steps=[('preprocessing', build_data_preprocessor(X, memory=cachedir1)),
                               ('feature_selector', SelectKBest(FEATURES_SELECTORS[fs_name], k=k)),
                               ('classifier', MODELS[estimator_name])],
                        memory=cachedir2)

    results = cross_validate(pipeline, X, y, cv=cv, scoring=get_scoring_metrics(y), return_estimator=True, verbose=2,
                             n_jobs=N_JOBS)

    base_log = {
        'learning_algorithm': estimator_name,
        'dataset': Path(filename).name,
        'filtering_algorithm': fs_name,
        'n_selected_features': k,
        'cv_method': str(cv),
        'n_samples': df.shape[0],
        'n_features_org': df.shape[1],
    }
    with open(log_filename, 'w') as f:
        pd.DataFrame(build_cv_logs(results, base_log)).to_csv(f)
    rmtree(cachedir1), rmtree(cachedir2)
    return log_filename


def build_cv_logs(results, base_log):
    logs = []
    for i in range(len(results['estimator'])):
        k_best, k_best_scores = extract_selected_features(results['estimator'][i])
        logs.append({
            **base_log,
            'fold': i + 1,
            'selected_features_names': ','.join([str(x) for x in k_best]),
            'selected_features_scores': ','.join(['%.4f' % x for x in k_best_scores]),
            **{k: '%.4f' % v[i] for k, v in results.items() if k != 'estimator'},
        })
    return logs


def extract_selected_features(estimator):
    fs_input_features = estimator['preprocessing'].get_feature_names_out()
    fs_scores = estimator['feature_selector'].scores_
    clf_input_features = estimator[:-1].get_feature_names_out()
    out_scores = {f: s for f, s in zip(fs_input_features, fs_scores) if f in clf_input_features}
    k_best, k_best_scores = zip(*sorted(out_scores.items(), key=lambda item: item[1], reverse=True))
    return k_best, k_best_scores


if __name__ == '__main__':
    run_all()
    # run_experiment('nb', 'data\\preprocessed\\arcene.csv', 'fdr', 1)
