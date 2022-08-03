import time

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from data_formatting import LABEL_COL


class DataPreprocessor:
    def __init__(self, y_nan_percent=0.1):
        """
        :param y_nan_percent: float, if the percentage of NaNs in y is more than that, we consider the NaN as an
                              additional class
        """
        self.imputer = SimpleImputer(strategy='mean')
        self.variance_thr = VarianceThreshold()
        self.normalizer = PowerTransformer()

        self.y_nan_percent = y_nan_percent
        self.y_nan_class = None

    def fit(self, x):
        """
        :param x: dataframe, where the last column is the target column
        :return:
        """
        assert x.columns[-1] == LABEL_COL, 'last column is not "y", check dataframe format'

        # handle missing labels
        if x[LABEL_COL].isna().mean() > self.y_nan_percent:
            self.y_nan_class = x[LABEL_COL].max() + 1

        X = x.drop(columns=[LABEL_COL])

        # categorical encoding
        if any(list(map(lambda c: not is_numeric_dtype(x[c]), x.columns))):
            raise NotImplementedError('Categorical values are not implemented yet')

        # imputing
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        # variance threshold
        X = self.variance_thr.fit_transform(X)

        # normalization
        self.normalizer.fit(X)

    def transform(self, x):
        assert x.columns[-1] == LABEL_COL, 'last column is not "y", check dataframe format'

        # handle missing labels
        if self.y_nan_class is None:
            x = x[~x[LABEL_COL].isna()]
        else:
            x[LABEL_COL].fillna(self.y_nan_class, inplace=True)

        X = x.drop(columns=[LABEL_COL])
        y = x[LABEL_COL].astype(int)

        # categorical encoding
        if any(list(map(lambda c: not is_numeric_dtype(x[c]), x.columns))):
            raise NotImplementedError('Categorical values are not implemented yet')

        # features imputation
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)

        # variance threshold
        X = self.variance_thr.transform(X)

        # normalization
        X = self.normalizer.transform(X)

        res = pd.DataFrame(X, columns=self.variance_thr.get_feature_names_out(), index=x.index)
        res[LABEL_COL] = y

        return res
