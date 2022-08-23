import os
import sys
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from itertools import product
import joblib
from run_experiments import get_dataset_and_experiment_params, build_log_dataframe
from disable_cv import DisabledCV
from experiments_settings import DATASETS_FILES, KS, N_JOBS, WRAPPED_MODELS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from data_preprocessor import build_data_preprocessor
from wrapped_estimators import WrappedSelectKBest
from wrapped_estimators.utils import get_cv
from feature_selectors import *
# from sklearnex import patch_sklearn
# patch_sklearn()

FEATURES_SELECTORS = [[select_fdr_fs, mrmr_fs, rfe_svm_fs, reliefF_fs],
                      [svm_fs], [svm_fs_New],
                      [rbf_svm_fs], [rbf_svm_fs_New],
                      [poly_svm_fs], [poly_svm_fs_New],
                      [grey_wolf_fs], [grey_wolf_fs_New]
                      ]
WRAPPED_FEATURES_SELECTORS = [[WrappedSelectKBest(score_func=joblib.Memory(mkdtemp(), verbose=0).cache(fs)) for fs in fss] for fss in FEATURES_SELECTORS]


def run_experiment(logs_dir='sbatch_logs', overwrite_logs=False):
    os.makedirs(logs_dir, exist_ok=True)
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    if len(sys.argv) == 1:
        datasets_files = DATASETS_FILES
    else:
        datasets_files = [name for arg in sys.argv[1:] for name in DATASETS_FILES if arg in name]
    feature_selectors, filename = list(product(WRAPPED_FEATURES_SELECTORS, datasets_files))[task_id]
    print(f'Start Experiment, Dataset: {filename}')
    dataset_name = Path(filename).name
    fs_name = feature_selectors[0].score_func.__name__ if len(feature_selectors) == 1 else 'baselines'
    log_filename = f'{dataset_name[:-len(".csv")]}_{fs_name}_results.csv'
    if logs_dir:
        log_filename = f'{logs_dir}/{log_filename}'

    # skip this experiment if exists
    if not overwrite_logs and Path(log_filename).exists():
        print('Exists, skipping')
        return log_filename

    X, y, cv, scoring = get_dataset_and_experiment_params(filename)

    cachedir = mkdtemp()
    pipeline = Pipeline(steps=[('dp', build_data_preprocessor(X)),
                               ('fs', 'passthrough'),
                               ('clf', 'passthrough')],
                        memory=cachedir)
    grid_params = {"fs": feature_selectors, "fs__k": KS, "clf": WRAPPED_MODELS}
    if isinstance(cv, StratifiedKFold):
        gcv = GridSearchCV(pipeline, grid_params, cv=cv, scoring=scoring, refit=False, verbose=2, n_jobs=N_JOBS)
        gcv.fit(X, y)
    else:
        gcv = GridSearchCV(pipeline, grid_params, cv=DisabledCV(), scoring=scoring, refit=False, verbose=2,
                           n_jobs=N_JOBS)
        gcv.fit(X, y, clf__cv=get_cv(X))
    res_df = build_log_dataframe(gcv, {'dataset': dataset_name,
                                       'n_samples': X.shape[0],
                                       'n_features_org': X.shape[1],
                                       'cv_method': str(cv)})
    res_df.to_csv(log_filename)

    rmtree(cachedir)
    print(f'Finished Experiment, Log file: {log_filename}')
    return log_filename


if __name__ == '__main__':
    run_experiment()
