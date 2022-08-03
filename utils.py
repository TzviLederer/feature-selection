from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

LABEL_COL = 'y'


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

        # fit features imputer
        self.imputer.fit(x.iloc[:, :-1])

        # fit target imputer
        if x[LABEL_COL].isna().mean() > self.y_nan_percent:
            self.y_nan_class = x[LABEL_COL].max() + 1

        # categorical encoding
        if any(list(map(lambda c: not is_numeric_dtype(x[c]), x.columns))):
            raise NotImplementedError('Categorical values are not implemented yet')

        # variance threshold
        x = self.variance_thr.fit_transform(x)

        # normalization
        x = self.normalizer.fit_transform(x)

    def transform(self, x):
        assert x.columns[-1] == LABEL_COL, 'last column is not "y", check dataframe format'

        # features imputation
        x.iloc[:, :-1] = self.imputer.transform(x.iloc[:, :-1])

        # target imputation
        if self.y_nan_class is None:
            x = x[~x[LABEL_COL].isna()]
        else:
            x[LABEL_COL].fillna(self.y_nan_class, inplace=True)
        indexes = x.index

        # variance threshold
        x = self.variance_thr.transform(x)

        # normalization
        x[:, :-1] = self.normalizer.transform(x)[:, :-1]

        x = pd.DataFrame(x, columns=self.variance_thr.get_feature_names_out(), index=indexes)
        x[LABEL_COL] = x[LABEL_COL].astype(int)
        return x
