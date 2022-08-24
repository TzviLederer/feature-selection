import os
import sys
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from disable_cv import DisabledCV
from experiments_settings import DATASETS_FILES, N_JOBS, OVERRIDE_LOGS, WRAPPED_FEATURES_SELECTORS, WRAPPED_MODELS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from data_preprocessor import build_data_preprocessor, DataPreprocessorWrapper
from imblearn.over_sampling import BorderlineSMOTE  # choose the least common samples to duplicate (could perform better
from imblearn.pipeline import Pipeline  # IMPORTANT SO THAT SMOTE (sampler) WILL RUN ONLY ON FIT (train)

from run_experiments import build_log_dataframe, get_dataset_and_experiment_params
from wrapped_estimators.utils import get_cv


def run_all(results_file_name, logs_dir='logs_aug', overwrite_logs=False):
    # execute 'run_experiment' over all dataset / datasets in run arguments
    os.makedirs(logs_dir, exist_ok=True)
    if len(sys.argv) == 1:
        datasets = list(pd.read_csv(results_file_name).dataset.unique())
        datasets_files = [name for arg in datasets for name in DATASETS_FILES if arg in name]
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]

    for dataset_file in datasets_files:
        print(f'Start Experiment, Dataset: {dataset_file}')
        output_log_file = run_experiment(dataset_file, results_file_name, logs_dir=logs_dir,
                                         overwrite_logs=overwrite_logs)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(filename, results_file_name, logs_dir='logs_aug', overwrite_logs=False):
    # execute a single augumentation experiment (part 4) with the best setting for this dataset according to ROC_AUC
    # results in the results file.
    dataset_name = Path(filename).name
    log_filename = f'{dataset_name[:-len(".csv")]}_aug_results.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    # extract best settings
    fs, clf, k = extract_best_settings_from_results(results_file_name, dataset_name)

    # build CV, scoring function, and X,y for the requested dataset
    X, y, cv, scoring = get_dataset_and_experiment_params(filename)

    # Build the PCA augmentation required at the assignment - union of:
    #   * original features
    #   * PCA with linear kernel
    #   * PCA with rbf kernel
    pca_aug = FeatureUnion([('identity', FunctionTransformer()),
                            ('pca_linear', KernelPCA(kernel='linear')),
                            ('pca_rbf', KernelPCA(kernel='rbf'))])

    cachedir = mkdtemp()
    pipeline = Pipeline(steps=[('dp', DataPreprocessorWrapper(build_data_preprocessor(X))),  # wrapped preprocessor (explained at the wrapper description, required for using with imblearn pipeline)
                               ('fs', 'passthrough'),  # use passthrough for original structure compatability
                               ('pca', pca_aug),  # PCA augmentation
                               ('smote', BorderlineSMOTE()),  # BorderlineSMOTE augmentation
                               ('clf', 'passthrough')],  # use passthrough for original structure compatability
                        memory=cachedir)
    grid_params = {"fs": [fs], "clf": [clf], "fs__k": [k]}  # set params to best settings
    if isinstance(cv, StratifiedKFold):
        # if SKF, use GridSearchCV "normally"
        gcv = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring, refit=False, verbose=2, n_jobs=N_JOBS)
        gcv.fit(X, y)
    else:
        # if LPO, use GridSearchCV with DisableCV (no folding) - explain in DisableCV description
        gcv = GridSearchCV(pipeline, grid_params, cv=DisabledCV(), scoring=scoring, refit=False, verbose=2,
                           n_jobs=N_JOBS)
        gcv.fit(X, y, clf__cv=get_cv(X))  # pass real CV to the classifier for custom LPO fitting
    # build log
    res_df = build_log_dataframe(gcv, {'dataset': dataset_name,
                                       'n_samples': X.shape[0],
                                       'n_features_org': X.shape[1],
                                       'cv_method': str(cv)})
    # add suffix
    res_df['filtering_algorithm'] = res_df['filtering_algorithm'].map(lambda x: x + '_Aug')
    # save
    res_df.to_csv(log_filename)

    rmtree(cachedir)
    return log_filename


def extract_best_settings_from_results(results_file_name, dataset_name):
    # extract best setting according to ROC_AUC value for the given dataset from the experiments results
    df = pd.read_csv(results_file_name)
    df = df[(df['dataset'] == dataset_name) & ~(df['filtering_algorithm'].str.endswith('_Aug'))]
    gc = df.groupby(['learning_algorithm', 'filtering_algorithm', 'n_selected_features']).mean(
        'test_roc_auc').reset_index()
    best_settings = gc.iloc[gc['test_roc_auc'].argmax()][[
        'learning_algorithm', 'filtering_algorithm', 'n_selected_features']]
    fs = next((x for x in WRAPPED_FEATURES_SELECTORS if x.score_func.__name__ == best_settings['filtering_algorithm']))
    clf = next((x for x in WRAPPED_MODELS if x.clf_name_ == best_settings['learning_algorithm']))
    k = best_settings['n_selected_features']
    return fs, clf, k


if __name__ == '__main__':
    run_all('all_exp_df.csv', overwrite_logs=OVERRIDE_LOGS)
