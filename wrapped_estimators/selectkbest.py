import inspect
from sklearn.feature_selection import SelectKBest
from wrapped_estimators.utils import fit_with_time


class WrappedSelectKBest(SelectKBest):
    def __init__(self, *args, **kwargs):
        SelectKBest.__init__(self, *args, **kwargs)
        self.org_fit = self.fit
        self.fit = self.fit_modified
        self.fit_time = None

    def fit_modified(self, X, y, **kwargs):
        return fit_with_time(self, X, y, **kwargs)


WrappedSelectKBest.__init__.__signature__ = inspect.signature(SelectKBest.__init__)
