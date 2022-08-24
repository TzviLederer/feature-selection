from pathlib import Path
from feature_selectors import *
from wrapped_estimators import *
from tempfile import mkdtemp
import joblib

# settings for 'run_experiments.py'

# our wrapped classifiers
WRAPPED_MODELS = [WrappedGaussianNB(),
                  WrappedRandomForestClassifier(),
                  WrappedKNeighborsClassifier(),
                  WrappedLogisticRegression(max_iter=10_000),
                  WrappedSVC(kernel='rbf', probability=True)]

# feature selectors
FEATURES_SELECTORS = [select_fdr_fs, mrmr_fs, rfe_svm_fs, reliefF_fs,
                      svm_fs, svm_fs_New,
                      rbf_svm_fs, rbf_svm_fs_New,
                      poly_svm_fs, poly_svm_fs_New,
                      grey_wolf_fs, grey_wolf_fs_New]

# feature selectors wrapped with our custom select-k-best and applied caching using joblib's Memory
WRAPPED_FEATURES_SELECTORS = [WrappedSelectKBest(score_func=joblib.Memory(mkdtemp(), verbose=0).cache(fs)) for fs in
                              FEATURES_SELECTORS]

# k values
KS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

# We assume processed datasets are CSV files saved at ./data/preprocessed
DATASETS_FILES = list(map(str, Path('data/preprocessed').glob('*.csv')))

# should or shouldn't skip experiments with existing results file
OVERRIDE_LOGS = False

# GridSearchCV parallelism
N_JOBS = 1
