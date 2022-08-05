import numpy as np
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import SelectFdr, RFE, SelectKBest
from sklearn.svm import SVR
from skfeature.function.similarity_based import reliefF


def mrmr(k):
    def mrmr_k(X, y):
        best_k_idx, scores, _ = MRMR.mrmr(X.to_numpy(), y.to_numpy(), n_selected_features=k)
        res = np.zeros_like(X.shape[1])
        res[best_k_idx] = scores
        return res

    return SelectKBest(mrmr_k, k=k)


def select_fdr_k(k): # need filter k?
    return SelectFdr(alpha=0.1)


def rfe_svm_k(k):
    return RFE(SVR(kernel='linear'), n_features_to_select=k)


def reliefF_k(k):
    def reliefF_raw(X, y):
        return reliefF.reliefF(X, y, mode='raw')
    return SelectKBest(reliefF_raw, k=k)
