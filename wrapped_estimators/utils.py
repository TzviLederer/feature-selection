import time
from collections import defaultdict
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import LeaveOneOut, LeavePOut, StratifiedKFold
from scoring_handlers import calculate_metrics


def fit_with_time(self, X, y, **kwargs):
    # used in wrapper estimators to measure fit time
    start = time.time()
    return_value = self.org_fit(X, y, **kwargs)
    self.fit_time = time.time() - start
    return return_value


def fit_for_leave_out(self, X, y, cv=None, **kwargs):
    # used in wrapper estimators for leave out CVs
    # to calculate metrics like roc auc over LOO/LPO
    # we perform cross validation while keeping the test y_proba and use all probas for metrics calculation at once.
    y_pred_proba, mean_fit_time, mean_inference_time = cross_val_predict_lpo(self, X, y, cv=cv)
    self.metrics = calculate_metrics(y, y_pred_proba, multi=(len(np.unique(y)) > 2))
    self.fit_time = mean_fit_time
    self.metrics['mean_inference_time'] = mean_inference_time
    return self


def cross_val_predict_lpo(pipeline, X, y, cv):
    # perform cross validation of leave p out cv
    # return the test y_proba (for metrics calculation), and mean fit time and inference time
    outputs = defaultdict(list)
    fit_times = []
    pred_proba_times = []
    for train_ind, val_ind in cv.split(X):
        x_train, x_val = X[train_ind], X[val_ind]
        y_train, y_val = y[train_ind], y[val_ind]
        t1 = time.time()
        pipeline.org_fit(x_train, y_train)
        t2 = time.time()
        preds = pipeline.predict_proba(x_val)
        t3 = time.time()
        for i, p in zip(val_ind, preds):
            outputs[i].append(p)
        fit_times.append(t2 - t1)
        pred_proba_times.append((t3 - t2) / len(x_val))
        pipeline = clone(pipeline)
    y_pred_proba = np.array([np.stack(v).mean(axis=0) for _, v in sorted(outputs.items(), key=lambda item: item[0])])
    mean_fit_time = np.array(fit_times).mean()
    mean_inference_time = np.array(pred_proba_times).mean()
    return y_pred_proba, mean_fit_time, mean_inference_time


def get_cv(X):
    # returns the appropriate CV according to the assignment requirements
    if len(X) < 50:
        return LeavePOut(2)
    elif 50 <= len(X) <= 100:
        return LeaveOneOut()
    elif 100 < len(X) < 1000:
        return StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    return StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
