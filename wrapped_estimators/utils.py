import time
from collections import defaultdict
from sklearn.base import clone

import numpy as np
from sklearn.model_selection import LeaveOneOut, LeavePOut

from scoring_handlers import calculate_metrics


def fit_with_time(self, X, y, **kwargs):
    start = time.time()
    return_value = self.org_fit(X, y, **kwargs)
    self.fit_time = time.time() - start
    return return_value


def fit_for_leave_out(self, X, y, **kwargs):
    cv = get_cv(X)
    start = time.time()
    y_pred_proba = cross_val_predict_lpo(self, X, y, cv=cv)
    self.metrics = calculate_metrics(y, y_pred_proba, multi=(len(np.unique(y)) > 2))
    self.fit_time = (time.time() - start) / cv.get_n_splits(X)
    return self


def cross_val_predict_lpo(pipeline, X, y, cv):
    outputs = defaultdict(list)
    for train_ind, val_ind in cv.split(X):
        x_train, x_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        cloned_pipeline = clone(pipeline)
        cloned_pipeline.org_fit(x_train, y_train)
        preds = cloned_pipeline.predict_proba(x_val)
        for i, p in zip(val_ind, preds):
            outputs[i].append(p)
    return np.array([np.stack(v).mean(axis=0) for _, v in sorted(outputs.items(), key=lambda item: item[0])])


def get_cv(X):
    if len(X) < 50:
        return LeavePOut(2)
    elif 50 <= len(X) <= 100:
        return LeaveOneOut()
    elif 100 < len(X) < 1000:
        return StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    return StratifiedKFold(n_splits=5, random_state=42, shuffle=True)