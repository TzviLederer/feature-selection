from pathlib import Path
from feature_selectors import *
from wrapped_estimators import *

WRAPPED_MODELS = [WrappedGaussianNB(),
                  WrappedRandomForestClassifier(),
                  WrappedKNeighborsClassifier(),
                  WrappedLogisticRegression(max_iter=10_000),
                  WrappedSVC(kernel='rbf', probability=True)]

FEATURES_SELECTORS = [select_fdr_fs, mrmr_fs, rfe_svm_fs, reliefF_fs,
                      svm_fs, svm_fs_New,
                      grey_wolf_fs, grey_wolf_fs_New]
WRAPPED_FEATURES_SELECTORS = [WrappedSelectKBest(score_func=fs) for fs in FEATURES_SELECTORS]

KS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

DATASETS_FILES = list(map(str, Path('data/preprocessed').glob('*.csv')))
OVERRIDE_LOGS = True

N_JOBS = -1
