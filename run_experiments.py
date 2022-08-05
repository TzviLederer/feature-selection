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
    preprocessor = DataPreprocessor()

    log_filename = f'{"_".join([estimator_name, Path(filename).name, filtering_algo, str(num_selected_features)])}.json'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    log_experiment_params = {
        'learning_algo': estimator_name,
        'dataset': Path(filename).name,
        'filtering_algo': filtering_algo,
        'n_selected_features': num_selected_features,
        'cv_method': str(cv),
        'n_samples': df.shape[0],
        'n_features_org': df.shape[1]
    }

    # preprocessing
    preprocessor.fit(df)
    df = preprocessor.transform(df)
    METRICS = METRICS_B if len(set(df[LABEL_COL])) == 2 else METRICS_M

    # feature selection
    fs_fit_time, k_best, fs_out_scores = get_k_best(df, features_selector, num_selected_features)

    # evaluate results
    mean_fit_time, probas, mean_pred_time = fit_estimator(cv, df, estimator, k_best)
    metrics = {metric: metric_func(df[LABEL_COL], probas) for metric, metric_func in METRICS.items()}
    log_outputs = {'fit_time': mean_fit_time,
                   'fs_fit_time': fs_fit_time,
                   'selected_features_names': k_best,
                   'selected_features_scores': fs_out_scores,
                   **log_experiment_params,
                   **metrics}
    with open(log_filename, 'w') as f:
        json.dump(log_outputs, f)

    return log_filename


def fit_estimator(cv, df, estimator, k_best):
    probas = []
    fit_times = []
    predict_time = .0
    for train_index, val_index in cv.split(df, df['y']):
        t0 = time.time()
        estimator.fit(df.iloc[train_index][k_best], df.iloc[train_index][LABEL_COL])
        fit_times.append(time.time() - t0)

        t0 = time.time()
        probas.append(estimator.predict_proba(df.iloc[val_index][k_best]))
        predict_time += (time.time() - t0)
    mean_fit_time = np.array(fit_times).mean()
    probas = np.concatenate(probas, axis=0)
    mean_pred_time = predict_time / len(df)
    return mean_fit_time, probas, mean_pred_time


def get_k_best(df, features_selector, k):
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
