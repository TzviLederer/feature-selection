import os
import sys
import time
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_formatting import LABEL_COL
from disable_cv import DisabledCV
from experiments_settings import DATASETS_FILES, KS, N_JOBS, OVERRIDE_LOGS, WRAPPED_FEATURES_SELECTORS, \
    WRAPPED_MODELS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from data_preprocessor import build_data_preprocessor
from scoring_handlers import get_scoring
from wrapped_estimators.utils import get_cv
# from sklearnex import patch_sklearn
# patch_sklearn()


def run_all(logs_dir='logs', overwrite_logs=False):
    # execute 'run_experiment' over all dataset / datasets in run arguments
    os.makedirs(logs_dir, exist_ok=True)
    if len(sys.argv) == 1:
        datasets_files = DATASETS_FILES
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]

    for dataset_file in datasets_files:
        print(f'Start Experiment, Dataset: {dataset_file}')
        output_log_file = run_experiment(dataset_file, logs_dir=logs_dir, overwrite_logs=overwrite_logs)
        print(f'Finished Experiment, Log file: {output_log_file}')


def run_experiment(filename, logs_dir=None, overwrite_logs=True):
    # execute a single dataset experiment (parts 2, 3)
    dataset_name = Path(filename).name
    log_filename = f'{dataset_name[:-len(".csv")]}_results_{int(time.time())}.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    # build CV, scoring function, and X,y for the requested dataset
    X, y, cv, scoring = get_dataset_and_experiment_params(filename)

    cachedir = mkdtemp()
    pipeline = Pipeline(steps=[('dp', build_data_preprocessor(X)),  # preprocessor
                               ('fs', 'passthrough'),  # feature selector - using passthrough to take it from grid
                               ('clf', 'passthrough')], # classifier - using passthrough to take it from grid
                        memory=cachedir)
    grid_params = {"fs": WRAPPED_FEATURES_SELECTORS, "fs__k": KS, "clf": WRAPPED_MODELS} # set grid for experiment
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
    # save
    res_df.to_csv(log_filename)

    rmtree(cachedir)
    return log_filename


def get_dataset_and_experiment_params(filename):
    # Given a path to CSV file, we load it, separate the features and label, label encode the label and build the
    # relevant CV according to the assignment requirements, and the scoring function according to it.
    df = pd.read_csv(filename)
    cv = get_cv(df)
    print(str(cv))
    # check if the number of sample in each class is less than fold number
    if isinstance(cv, StratifiedKFold):
        vc = df[LABEL_COL].value_counts()
        df = df[df[LABEL_COL].isin(vc[vc > cv.n_splits].index)]
    X = df.drop(columns=[LABEL_COL])
    y = pd.Series(LabelEncoder().fit_transform(df[LABEL_COL]))
    return X, y, cv, get_scoring(cv, y)


def build_log_dataframe(gcv, base_details):
    # This function extracts all relevant details fro the GridSearchCV results for logging in the assignment format,
    # and based on our custom scoring formats.
    to_log = []
    for j, experiment in enumerate(gcv.cv_results_['params']):
        for i in range(gcv.n_splits_):
            fold_res = {k[len(f'split{i}_'):]: v[j] for k, v in gcv.cv_results_.items() if k.startswith(f'split{i}_')}
            # selected features probabilities are passed through scoring by using 'feature_prob' suffix.
            sf = {k[len('test_'):-len('_feature_prob')]: v for k, v in fold_res.items() if
                  k.endswith('_feature_prob') and v > 0}
            sf = dict(sorted(sf.items(), key=lambda item: item[1], reverse=True))
            fold_res = {k: v for k, v in fold_res.items() if not k.endswith('_feature_prob')}
            to_log.append({**fold_res,
                           **base_details,
                           'learning_algorithm': experiment['clf'].clf_name_,
                           'filtering_algorithm': experiment['fs'].score_func.__name__,
                           'n_selected_features': experiment['fs__k'],
                           'selected_features_names': ','.join([str(x) for x in sf.keys()]),
                           'selected_features_scores': ','.join(['%.4f' % x for x in sf.values()]),
                           })
    return pd.DataFrame(to_log)


if __name__ == '__main__':
    run_all(overwrite_logs=OVERRIDE_LOGS)
