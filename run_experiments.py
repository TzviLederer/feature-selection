import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data_formatting import LABEL_COL
from data_preprocessor import DataPreprocessor
from experiments_settings import METRICS_B, METRICS_M, DATASETS_FILES, FEATURES_SELECTORS, MODELS, KS
from utils import get_cv


def run_all(logs_dir='logs', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    for experiment_args in product(MODELS.keys(), DATASETS_FILES, FEATURES_SELECTORS.keys(), KS):
        print(f'Start Experiment, Settings: {experiment_args}')
        output_log_file = run_experiment(*experiment_args, logs_dir=logs_dir, overwrite_logs=overwrite_logs)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(estimator_name, filename, filtering_algo, num_selected_features, logs_dir=None, overwrite_logs=True):
    estimator = MODELS[estimator_name]
    df = pd.read_csv(filename)
    cv = get_cv(df)
    features_selector = FEATURES_SELECTORS[filtering_algo](num_selected_features)
    METRICS = METRICS_B if len(set(df[LABEL_COL])) == 2 else METRICS_M

    log_filename = f'{"_".join([estimator_name, Path(filename).name, filtering_algo, str(num_selected_features)])}.json'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    exp_logs = []
    for train_index, val_index in cv.split(df, df['y']):
        metrics, stats = train_and_evaluate(estimator, features_selector, num_selected_features, df[train_index],
                                            df[val_index], METRICS)

        base_log = {
            'learning_algo': estimator_name,
            'dataset': Path(filename).name,
            'filtering_algo': filtering_algo,
            'n_selected_features': num_selected_features,
            'cv_method': str(cv),
            'n_samples': df.shape[0],
            'n_features_org': df.shape[1],
            **stats
        }
        exp_logs.extend([{**base_log, metric: value} for metric, value in metrics.items()])

    with open(log_filename, 'w') as f:
        json.dump(exp_logs, f)

    return log_filename


def train_and_evaluate(estimator, features_selector, num_selected_features, train_df, val_df, metrics_funcs):
    # preprocessing
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)
    train_df = preprocessor.transform(train_df)
    val_df = preprocessor.transform(val_df)
    # feature selection
    fs_time, k_best, k_best_scores = get_k_best_features(train_df, features_selector, num_selected_features)
    # fit estimator
    t0 = time.time()
    estimator.fit(train_df[k_best], train_df[LABEL_COL])
    fit_time = time.time() - t0
    # evaluate
    t0 = time.time()
    probas = estimator.predict_proba(val_df[k_best])
    predict_time = (time.time() - t0)
    metrics = {metric: func(val_df[LABEL_COL], probas) for metric, func in metrics_funcs.items()}
    return metrics, {'fit_time': fit_time,
                     'feature_selection_time': fs_time,
                     'predict_time': predict_time,
                     'selected_features_names': k_best,
                     'selected_features_scores': k_best_scores}


def get_k_best_features(df, features_selector, k):
    X, y = df.drop(LABEL_COL, axis=1), df[LABEL_COL]
    # feature selection fit
    start_time = time.time()
    features_selector.fit(X, y)
    fit_time = time.time() - start_time

    # selected features scores
    fs_out = features_selector.get_feature_names_out()
    out_scores = {f: s for f, s in zip(features_selector.feature_names_in_, features_selector.scores_) if k in fs_out}
    k_best, k_best_scores = zip(*sorted(out_scores.items(), key=lambda item: item[1], reverse=True)[:k])
    return fit_time, k_best, k_best_scores


if __name__ == '__main__':
    run_all()
