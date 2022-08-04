import numpy as np
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold, StratifiedKFold

from data_formatting import LABEL_COL


def _pr_auc(y_true, y_score):
    # https://sinyi-chou.github.io/python-sklearn-precision-recall/
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def pr_auc(y_true, y_score):
    classes = np.unique(y_true)
    n_classes = len(classes)
    if n_classes > 2:
        y_true_bin = label_binarize(y_true, classes=classes)
        return np.mean([_pr_auc(y_true_bin[:, i], y_score[:, i]) for i in range(n_classes)])
    else:
        return _pr_auc(y_true, y_score)


def get_cv(df):
    if len(df) < 50:
        return LeavePOut(2)
    elif 50 <= len(df) <= 100:
        return LeaveOneOut()
    elif 100 < len(df) < 1000:
        return StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    return StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


def _needs_proba(metric: str):
    if metric in {'acc', 'mcc'}:
        return False
    elif metric in {'roc_auc', 'pr_auc'}:
        return True
    raise NotImplementedError(f'You must determine if this score ({metric}) does need proba')


def calculate_metrics_scores(estimator, df_val, metrics_dict):
    return {k: make_scorer(metric, needs_proba=_needs_proba(k))(estimator, df_val.drop(LABEL_COL, axis=1), df_val[LABEL_COL])
            for k, metric in metrics_dict.items()}
