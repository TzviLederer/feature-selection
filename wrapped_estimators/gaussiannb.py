import inspect
from sklearn.naive_bayes import GaussianNB
from .utils import fit_with_time, fit_for_leave_out


class WrappedGaussianNB(GaussianNB):
    def __init__(self, *args, **kwargs):
        GaussianNB.__init__(self, *args, **kwargs)
        self.org_fit = self.fit
        self.fit = self.fit_modified
        self.fit_time = None
        self.metrics = {}
        self.clf_name_ = 'GaussianNB'

    def fit_modified(self, X, y, leave_out_mode=False, **kwargs):
        return fit_for_leave_out(self, X, y, **kwargs) if leave_out_mode else fit_with_time(self, X, y, **kwargs)


WrappedGaussianNB.__init__.__signature__ = inspect.signature(GaussianNB.__init__)
