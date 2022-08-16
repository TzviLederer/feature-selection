import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize


def _pr_auc(y_true, y_score):
    # https://sinyi-chou.github.io/python-sklearn-precision-recall/
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def pr_auc(y_true, y_score):
    classes = np.unique(y_true)
    n_classes = len(classes)
    if n_classes <= 2:
        return _pr_auc(y_true, y_score[:, 1])
    y_true_bin = label_binarize(y_true, classes=classes)
    return np.mean([_pr_auc(y_true_bin[:, i], y_score[:, i]) for i in range(n_classes)])
