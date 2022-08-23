import inspect
from sklearn.neighbors import KNeighborsClassifier
from .utils import fit_with_time, fit_for_leave_out


class WrappedKNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, *args, **kwargs):
        KNeighborsClassifier.__init__(self, *args, **kwargs)
        self.org_fit = self.fit
        self.fit = self.fit_modified
        self.fit_time = None
        self.metrics = {}
        self.clf_name_ = 'KNeighborsClassifier'

    def fit_modified(self, X, y, leave_out_mode=False, **kwargs):
        return fit_for_leave_out(self, X, y, cv=leave_out_mode, **kwargs) if leave_out_mode else fit_with_time(self, X, y, **kwargs)


WrappedKNeighborsClassifier.__init__.__signature__ = inspect.signature(KNeighborsClassifier.__init__)
