import time
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFdr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, make_scorer, precision_recall_curve, auc
from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from utils import DataPreprocessor

LABEL_COL = 'y'


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
        return KFold(n_splits=10)
    return KFold(n_splits=5)


def main():
    models = {'nb': GaussianNB(),
              'svm': SVC(kernel='rbf'),
              'lr': LogisticRegression(),
              'rf': RandomForestClassifier(),
              'knn': KNeighborsClassifier()}

    metrics = {'roc': roc_auc_score,
               'acc': accuracy_score,
               'mcc': matthews_corrcoef,
               'pr_auc': pr_auc}

    features_selectors = {'fdr': SelectFdr(alpha=0.1)}

    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

    filename = 'data/preprocessed/arcene.csv'
    output_log_filename = 'log_out.csv'
    filtering_algo = 'fdr'
    estimator_name = 'lr'
    k_ind = 5

    estimator = models[estimator_name]
    df = pd.read_csv(filename)
    cv = get_cv(df)
    scoring = {k: make_scorer(val) for k, val in metrics.items()}
    preprocessor = DataPreprocessor()
    features_selector = features_selectors[filtering_algo]
    k = ks[k_ind]

    outputs = []
    for i, (train_index, val_index) in enumerate(cv.split(df)):
        out_i = one_fold_pipe(df, estimator, features_selector, k, preprocessor, scoring, train_index, val_index)
        log_i = {'fit_time': out_i[0], 'fold': i}
        log_i.update(out_i[1])
        log_i['selected_features_names'] = list(out_i[2].keys())
        log_i['selected_features_scores'] = list(out_i[2].values())
        outputs.append(log_i)

    log_table = generate_log_table(cv, df, estimator_name, filename, filtering_algo, k, metrics, outputs)
    log_table.to_csv(output_log_filename)


def generate_log_table(cv, df, estimator_name, filename, filtering_algo, k, metrics, outputs):
    out_df = pd.DataFrame(outputs)
    out_df['dataset'] = Path(filename).name
    out_df['n_samples'] = df.shape[0]
    out_df['n_features_org'] = df.shape[1]
    out_df['filtering_algo'] = filtering_algo
    out_df['learning_algo'] = estimator_name
    out_df['n_selected_features'] = k
    out_df['cv_method'] = str(cv)
    out_df = pd.DataFrame([arrange_metrics(i, metric, out_df) for i, metric in product(out_df.index, metrics.keys())])
    out_df = out_df.drop(metrics.keys(), axis=1)
    return out_df


def arrange_metrics(i, metric, out_df):
    row = out_df.loc[i].copy()
    row['metric'] = metric
    row['metric_val'] = row[metric]
    return row


def one_fold_pipe(df, estimator, features_selector, k, preprocessor, scoring, train_index, val_index):
    # split df
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]

    # preprocessing
    preprocessor.fit(df_train)
    df_train = preprocessor.transform(df_train)

    # feature selection fit
    values_train = features_selector.fit_transform(df_train.drop(LABEL_COL, axis=1), df_train[LABEL_COL])
    df_train = pd.DataFrame(values_train, columns=features_selector.get_feature_names_out(), index=df_train.index)
    df_train[LABEL_COL] = df.iloc[train_index, -1]

    # preprocessing validation
    df_val = preprocessor.transform(df_val)
    values_val = features_selector.transform(df_val.drop(LABEL_COL, axis=1))
    df_val = pd.DataFrame(values_val, columns=features_selector.get_feature_names_out(), index=df_val.index)
    df_val[LABEL_COL] = df.iloc[val_index, -1]

    # select features
    scores = dict(zip(features_selector.feature_names_in_, features_selector.scores_))
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    scores = dict(list(scores.items())[:k])
    features = list(scores.keys())

    df_train = df_train[features + [LABEL_COL]]
    df_val = df_val[features + [LABEL_COL]]

    # fit
    t0_fit = time.time()
    estimator.fit(df_train.drop(LABEL_COL, axis=1), df_train[LABEL_COL])
    fit_time = time.time() - t0_fit

    # eval
    metrics_scores = {k: scorer(estimator, df_val.drop(LABEL_COL, axis=1), df_val[LABEL_COL])
                      for k, scorer in scoring.items()}
    return fit_time, metrics_scores, scores


if __name__ == '__main__':
    main()
