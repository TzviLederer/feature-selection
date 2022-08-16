import time
from collections import defaultdict

import numpy as np
from sklearn.model_selection import LeaveOneOut, LeavePOut

from scoring_handlers import calculate_metrics


def fit_with_time(self, X, y, **kwargs):
    start = time.time()
    return_value = self.org_fit(X, y, **kwargs)
    self.fit_time = time.time() - start
    return return_value


def fit_for_leave_out(self, X, y, **kwargs):
    cv = LeaveOneOut() if len(X) < 50 else LeavePOut(p=2)
    start = time.time()
    y_pred_proba = cross_val_predict_lpo(self, X, y, cv=cv)
    self.metrics = calculate_metrics(y, y_pred_proba, multi=(len(np.unique(y)) > 2))
    self.fit_time = time.time() - start
    return self


def cross_val_predict_lpo(pipeline, X, y, cv):
    outputs = defaultdict(list)
    for train_ind, val_ind in cv.split(X):
        x_train, x_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        pipeline.org_fit(x_train, y_train)
        preds = pipeline.predict_proba(x_val)
        for i, p in zip(val_ind, preds):
            outputs[i].append(p)
    return np.array([np.stack(v).mean(axis=0) for _, v in sorted(outputs.items(), key=lambda item: item[0])])
