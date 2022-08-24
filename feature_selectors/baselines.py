import numpy as np
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from sklearn.feature_selection import SelectFdr, RFE
from sklearn.svm import SVR


def mrmr_fs(X, y):
    # score function of select mRMR
    best_k_idx, scores, _ = MRMR.mrmr(X, y, n_selected_features=100)
    res = np.zeros(X.shape[1])
    res[best_k_idx] = scores
    return res


def select_fdr_fs(X, y):
    # score function of select fdr
    fs = SelectFdr(alpha=0.1)
    fs.fit(X, y)
    return fs.get_support().astype(int) * fs.scores_


def rfe_svm_fs(X, y):
    # score function of rfe svm
    fs = RFE(SVR(kernel='linear', max_iter=100_000), n_features_to_select=100)
    fs.fit(X, y)
    return fs.get_support().astype(int)


def reliefF_fs(X, y):
    # score function of reliefF
    return reliefF.reliefF(X, y, mode='raw')
