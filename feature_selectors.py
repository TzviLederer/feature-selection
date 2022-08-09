import numpy as np
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.similarity_based import reliefF
from sklearn.feature_selection import SelectFdr, RFE
from sklearn.svm import SVR, SVC


def mrmr_fs(X, y):
    best_k_idx, scores, _ = MRMR.mrmr(X, y, n_selected_features=100)
    res = np.zeros(X.shape[1])
    res[best_k_idx] = scores
    return res


def select_fdr_fs(X, y):
    fs = SelectFdr(alpha=0.1)
    fs.fit(X, y)
    return fs.get_support().astype(int) * fs.scores_


def rfe_svm_fs(X, y):
    fs = RFE(SVR(kernel='linear', max_iter=100_000), n_features_to_select=100)
    fs.fit(X, y)
    return fs.get_support().astype(int)


def reliefF_fs(X, y):
    return reliefF.reliefF(X, y, mode='raw')


def svm_fs(X, y, svm_max_iter=10_000_000, kernel='linear', verbose=0):
    X = np.array(X)
    y = np.array(y)

    svm = SVC(kernel=kernel, max_iter=svm_max_iter)

    X_0 = X.copy()
    s = list(range(X.shape[1]))
    r = []

    while s:
        if verbose > 0:
            print('\r  ', end='')
            print(f'\r{len(s)}', end='')

        svm.fit(X_0[:, s], y)

        alphas = np.zeros(len(X))
        alphas[svm.support_] = svm.dual_coef_.mean(axis=0)
        w = alphas @ X_0[:, s]
        c = w ** 2

        f = np.argmin(c)
        r.append(s[f])
        s.remove(s[f])

    r = np.array(r)[::-1]

    # make scores
    t = np.array(list(dict(sorted(enumerate(r), key=lambda x: x[1])).keys()))
    return 1 - t / max(t)

# FEATURES_SELECTORS = [SelectKBest(fs) for fs in [mrmr_fs, select_fdr_fs, rfe_svm_fs, reliefF_fs]]
# FEATURES_SELECTORS = [SelectKBest(fs) for fs in [svm_fs]]
