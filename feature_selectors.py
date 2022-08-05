import numpy as np
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import SelectFdr, RFE
from sklearn.svm import SVR
from skfeature.function.similarity_based import reliefF

from data_formatting import LABEL_COL


def get_best_k_features(feature_names, scores, k):
    best_k_idx = np.argsort(scores, axis=0)[::-1][:k]

    k_best = feature_names[best_k_idx]
    k_best_scores = scores[best_k_idx]
    k_best, k_best_scores = zip(*sorted(list(zip(k_best, k_best_scores)), key=lambda item: item[1], reverse=True))
    return k_best, k_best_scores


def mrmr_feature_selector(df, k):
    X, y = df.drop(LABEL_COL, axis=1), df[LABEL_COL]
    best_k_idx, scores, _ = MRMR.mrmr(X.to_numpy(), y.to_numpy(), n_selected_features=k)
    return get_best_k_features(X.columns[best_k_idx], scores, k)


def select_fdr_feature_selector(df, k):
    X, y = df.drop(LABEL_COL, axis=1), df[LABEL_COL]
    fs = SelectFdr(alpha=0.1)
    fs.fit(X, y)
    feature_names, scores = zip(*[(f, s) for f, s in zip(fs.feature_names_in_, fs.scores_)
                                  if f in fs.get_feature_names_out()])
    return get_best_k_features(feature_names, scores, k)


def rfe_svm(df, k):
    X, y = df.drop(LABEL_COL, axis=1), df[LABEL_COL]
    fs = RFE(SVR(kernel='linear'), n_features_to_select=k)


def reliefF_feature_selector(df, k):
    X, y = df.drop(LABEL_COL, axis=1), df[LABEL_COL]
    scores = reliefF.reliefF(X.to_numpy(), y.to_numpy())
    return get_best_k_features(X.columns, scores, k)
