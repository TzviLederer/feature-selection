import numpy as np
from sklearn.svm import SVC


def svm_fs(X, y, svm_max_iter=10_000_000, kernel='linear'):
    X = np.array(X)
    y = np.array(y)

    svm = SVC(kernel=kernel, max_iter=svm_max_iter)

    X_0 = X.copy()
    s = list(range(X.shape[1]))
    r = []

    while s:
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


def rbf_svm_fs(X, y, svm_max_iter=10_000_000):
    return svm_fs(X, y, svm_max_iter=svm_max_iter, kernel='rbf')


def poly_svm_fs(X, y, svm_max_iter=10_000_000):
    return svm_fs(X, y, svm_max_iter=svm_max_iter, kernel='poly')


def svm_fs_New(X, y, svm_max_iter=10_000_000, kernel='linear', step_frac=0.1):
    X = np.array(X)
    y = np.array(y)

    svm = SVC(kernel=kernel, max_iter=svm_max_iter)

    X_0 = X.copy()
    s = list(range(X.shape[1]))
    r = []

    while s:
        svm.fit(X_0[:, s], y)

        alphas = np.zeros(len(X))
        alphas[svm.support_] = svm.dual_coef_.mean(axis=0)
        w = alphas @ X_0[:, s]
        c = w ** 2

        f = np.argsort(c)[:1 + int(len(c) * step_frac)]
        for f_i in f:
            r.append(s[f_i])
        s = [x for idx, x in enumerate(s) if idx not in f]

    r = np.array(r)[::-1]

    # make scores
    t = np.array(list(dict(sorted(enumerate(r), key=lambda x: x[1])).keys()))
    return 1 - t / max(t)


def rbf_svm_fs_New(X, y, svm_max_iter=10_000_000):
    return svm_fs_New(X, y, svm_max_iter=svm_max_iter, kernel='rbf')


def poly_svm_fs_New(X, y, svm_max_iter=10_000_000):
    return svm_fs_New(X, y, svm_max_iter=svm_max_iter, kernel='poly')
