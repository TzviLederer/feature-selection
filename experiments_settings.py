from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from feature_selectors import select_fdr_fs, mrmr_fs, rfe_svm_fs, reliefF_fs
from pr_auc import pr_auc

MODELS = {'nb': GaussianNB(),
          'svm': SVC(kernel='rbf', probability=True),
          'lr': LogisticRegression(max_iter=10_000),
          'rf': RandomForestClassifier(),
          'knn': KNeighborsClassifier()}

FEATURES_SELECTORS = {'fdr': select_fdr_fs,
                      'mrmr': mrmr_fs,
                      'rfe_svm': rfe_svm_fs,
                      'reliefF': reliefF_fs}

KS = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

DATASETS_FILES = list(map(str, Path('data/preprocessed').glob('*.csv')))

# binary and multiclass metrics
METRICS_B = {'roc_auc': 'roc_auc',
             'acc': 'accuracy',
             'mcc': make_scorer(matthews_corrcoef),
             'pr_auc': make_scorer(pr_auc, needs_proba=True)}

METRICS_M = {'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr', needs_proba=True),
             'acc': 'accuracy',
             'mcc': make_scorer(matthews_corrcoef),
             'pr_auc': make_scorer(pr_auc, needs_proba=True)}

# Scores for leave-one-out or leave-two-out
SCORES_B = {'roc_auc': lambda y_true, y_score: roc_auc_score(y_true, y_score[:, 1]),
            'acc': lambda y_true, y_score: accuracy_score(y_true, y_score.argmax(axis=1)),
            'mcc': lambda y_true, y_score: matthews_corrcoef(y_true, y_score.argmax(axis=1)),
            'pr_auc': lambda y_true, y_score: pr_auc(y_true, y_score[:, 1])
            }

SCORES_M = {}

N_JOBS = -1


def get_cv(X):
    if len(X) < 50:
        return LeavePOut(2)
    elif 50 <= len(X) <= 100:
        return LeaveOneOut()
    elif 100 < len(X) < 1000:
        return StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    return StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


def get_scoring_metrics(y):
    return METRICS_B if len(y.unique()) == 2 else METRICS_M


def get_scores_for_loo(y):
    return SCORES_B if len(y.unique()) == 2 else SCORES_M
