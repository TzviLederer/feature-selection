import json
import os
import sys
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from disable_cv import DisabledCV
from experiments_settings import DATASETS_FILES, N_JOBS, OVERRIDE_LOGS, WRAPPED_FEATURES_SELECTORS, WRAPPED_MODELS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from data_preprocessor import build_data_preprocessor
from imblearn.over_sampling import BorderlineSMOTE  # choose the least common samples to duplicate (could perform better
from imblearn.pipeline import Pipeline  # IMPORTANT SO THAT SMOTE (sampler) WILL RUN ONLY ON FIT (train)

from run_aug_experiment import DataPreprocessorWrapper
from run_experiments import build_log_dataframe, get_dataset_and_experiment_params


def run_all(results_file_name, logs_dir='logs_aug', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    if len(sys.argv) == 1:
        datasets_files = DATASETS_FILES
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]
    dataset_file = datasets_files[task_id]
    print(f'Start Experiment, Dataset: {dataset_file}')
    output_log_file = run_experiment(dataset_file, results_file_name, logs_dir=logs_dir,
                                     overwrite_logs=overwrite_logs)
    print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(filename, results_file_name, logs_dir='logs_aug', overwrite_logs=False):
    dataset_name = Path(filename).name
    log_filename = f'{dataset_name[:-len(".csv")]}_aug_results.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    fs, clf, k = extract_best_settings_from_results(results_file_name, dataset_name)

    X, y, cv, scoring = get_dataset_and_experiment_params(filename)
    pca_aug = FeatureUnion([('identity', FunctionTransformer()),
                            ('pca_linear', KernelPCA(kernel='linear')),
                            ('pca_rbf', KernelPCA(kernel='rbf'))])

    cachedir = mkdtemp()
    pipeline = Pipeline(steps=[('dp', DataPreprocessorWrapper(build_data_preprocessor(X))),
                               ('fs', 'passthrough'),
                               ('pca', pca_aug),
                               ('smote', BorderlineSMOTE()),
                               ('clf', 'passthrough')],
                        memory=cachedir)
    grid_params = {"fs": [fs], "clf": [clf], "fs__k": [k]}
    if isinstance(cv, StratifiedKFold):
        gcv = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring, refit=False, verbose=2, n_jobs=N_JOBS)
        gcv.fit(X, y)
    else:
        gcv = GridSearchCV(pipeline, grid_params, cv=DisabledCV(), scoring=scoring, refit=False, verbose=2,
                           n_jobs=N_JOBS)
        gcv.fit(X, y, clf__leave_out_mode=True)
    res_df = build_log_dataframe(gcv, {'dataset': dataset_name,
                                       'n_samples': X.shape[0],
                                       'n_features_org': X.shape[1],
                                       'cv_method': str(cv)})
    res_df['learning_algorithm'] = res_df['learning_algorithm'].map(lambda x: x + '_Aug')
    res_df.to_csv(log_filename)

    rmtree(cachedir)
    return log_filename


def extract_best_settings_from_results(results_file_name, dataset_name):
    df = pd.read_csv(results_file_name)
    df = df[(df['dataset'] == dataset_name)]
    gc = df.groupby(['learning_algorithm', 'filtering_algorithm', 'n_selected_features']).mean(
        'test_roc_auc').reset_index()
    best_settings = gc.iloc[gc['test_roc_auc'].argmax()][[
        'learning_algorithm', 'filtering_algorithm', 'n_selected_features']]
    fs = next((x for x in WRAPPED_FEATURES_SELECTORS if x.score_func.__name__ == best_settings['filtering_algorithm']))
    clf = next((x for x in WRAPPED_MODELS if x.clf_name_ == best_settings['learning_algorithm']))
    k = best_settings['n_selected_features']
    return fs, clf, k


if __name__ == '__main__':
    run_all('unified_df.csv', overwrite_logs=OVERRIDE_LOGS)
