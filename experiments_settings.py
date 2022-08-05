from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, make_scorer

from feature_selectors import select_fdr_k, mrmr_k, rfe_svm_k, reliefF_k
from utils import pr_auc

MODELS = {'nb': GaussianNB(),
          'svm': SVC(kernel='rbf'),
          'lr': LogisticRegression(max_iter=10_000),
          'rf': RandomForestClassifier(),
          'knn': KNeighborsClassifier()}

FEATURES_SELECTORS = {'fdr': select_fdr_k,
                      'mrmr': mrmr_k,
                      'rfe_svm': rfe_svm_k,
                      'reliefF': reliefF_k}

KS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

DATASETS_FILES = list(map(str, Path('data/preprocessed').glob('*.csv')))

# binary and multiclass metrics
METRICS_B = {'roc_auc': lambda y_true, y_score: roc_auc_score(y_true, y_score[:, 1]),
             'acc': lambda y_true, y_score: accuracy_score(y_true, np.argmax(y_score, axis=1)),
             'mcc': lambda y_true, y_score: matthews_corrcoef(y_true, np.argmax(y_score, axis=1)),
             'pr_auc': lambda y_true, y_score: pr_auc(y_true, y_score[:, 1])}

METRICS_M = {'roc_auc': lambda y_true, y_score: roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr'),
             'acc': lambda y_true, y_score: accuracy_score(y_true, np.argmax(y_score, axis=1)),
             'mcc': lambda y_true, y_score: matthews_corrcoef(y_true, np.argmax(y_score, axis=1)),
             'pr_auc': pr_auc}
