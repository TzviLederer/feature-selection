import numpy as np


class DisabledCV:
    # https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    # Fake CV used to disable CV in GridSearchCV (required for our implementation).
    # Used when CV is LPO because we customly perform the cross validation of LPO inside the classifier's
    # fit function - therefore we disable outer CV so that inner CV will get the whole dataset.
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield np.arange(len(X)), np.arange(len(y))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
