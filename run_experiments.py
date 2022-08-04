import os
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import SelectKBest

from data_preprocessor import DataPreprocessor
from data_formatting import LABEL_COL
from experiments_settings import METRICS, DATASETS_FILES, FEATURES_SELECTORS, MODELS, KS
from utils import get_cv, calculate_metrics_scores


def run_all(logs_dir='logs'):
    os.makedirs(logs_dir, exist_ok=True)
    for experiment_args in zip(MODELS.keys(), DATASETS_FILES, FEATURES_SELECTORS.keys(), KS):
        print(f'Start Experiment, Settings: {experiment_args}')
        output_log_file = run_experiment(*experiment_args, logs_dir=logs_dir)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(estimator_name, filename, filtering_algo, num_selected_features, logs_dir=None):
    estimator = MODELS[estimator_name]
    df = pd.read_csv(filename)
    cv = get_cv(df)
    features_selector = FEATURES_SELECTORS[filtering_algo]
    preprocessor = DataPreprocessor()

    log_filename = f'{"_".join([estimator_name, Path(filename).name, filtering_algo, str(num_selected_features)])}.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    log_experiment_params = {
        'learning_algo': estimator_name,
        'dataset': Path(filename).name,
        'filtering_algo': filtering_algo,
        'n_selected_features': num_selected_features,
        'cv_method': str(cv),
        'n_samples': df.shape[0],
        'n_features_org': df.shape[1]
    }

    outputs = []
    for i, (train_index, val_index) in enumerate(cv.split(df)):
        fs_fit_time, fit_time, metrics_scores, fs_out_scores = one_fold_pipe(df, estimator, features_selector, num_selected_features,
                                                           preprocessor, train_index, val_index)
        print(f'Fold {i}, fs_fit: {fs_fit_time} secs, fit: {fit_time} secs, Results:')
        print(metrics_scores)
        for metric, metric_val in metrics_scores.items():
            outputs.append({'fit_time': fit_time,
                            'fs_fit_time': fs_fit_time,
                            'fold': i,
                            'selected_features_names': list(fs_out_scores.keys()),
                            'selected_features_scores': list(fs_out_scores.values()),
                            'metric': metric,
                            'metric_val': metric_val,
                            **log_experiment_params})
    pd.DataFrame(outputs).to_csv(log_filename)
    return log_filename


def one_fold_pipe(df, estimator, features_selector, num_selected_features, preprocessor, train_index, val_index):
    # split df
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    # preprocessing
    preprocessor.fit(df_train)
    df_train = preprocessor.transform(df_train)
    df_val = preprocessor.transform(df_val)

    # feature selection fit
    fs_fit_start_time = time.time()
    features_selector.fit(df_train.drop(LABEL_COL, axis=1), df_train[LABEL_COL])
    fs_fit_time = time.time() - fs_fit_start_time

    # selected features scores
    out_fs = features_selector.get_feature_names_out()
    fs_out_scores = {k: v for k, v in zip(features_selector.feature_names_in_, features_selector.scores_)
                     if k in out_fs}
    fs_out_scores = dict(sorted(fs_out_scores.items(), key=lambda item: item[1], reverse=True))

    # feature selection transform
    values_train = features_selector.transform(df_train.drop(LABEL_COL, axis=1))
    df_train = pd.DataFrame(values_train, columns=out_fs, index=df_train.index)
    df_train[LABEL_COL] = df.iloc[train_index][LABEL_COL]

    values_val = features_selector.transform(df_val.drop(LABEL_COL, axis=1))
    df_val = pd.DataFrame(values_val, columns=out_fs, index=df_val.index)
    df_val[LABEL_COL] = df.iloc[val_index][LABEL_COL]

    # fit model
    fit_start_time = time.time()
    estimator.fit(df_train.drop(LABEL_COL, axis=1), df_train[LABEL_COL])
    fit_time = time.time() - fit_start_time

    # evaluate
    metrics_scores = calculate_metrics_scores(estimator, df_val, METRICS)
    return fs_fit_time, fit_time, metrics_scores, fs_out_scores


if __name__ == '__main__':
    run_experiment(estimator_name='lr',
                   filename='data/preprocessed/arcene.csv',
                   filtering_algo='fdr',
                   num_selected_features=5)
