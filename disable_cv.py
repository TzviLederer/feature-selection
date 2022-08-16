import numpy as np


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield np.arange(len(X)), np.arange(len(y))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
