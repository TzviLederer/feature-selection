import time

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, LabelEncoder
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
        self.le = {}

        self.y_nan_percent = y_nan_percent
        self.y_nan_class = None

    def fit(self, x):
        """
        :param x: dataframe, where the last column is the target column
        :return:
        """
        assert LABEL_COL in x.columns, f'label column "{LABEL_COL}" is not in dataframe, check dataframe format'

        # handle missing labels
        if x[LABEL_COL].isna().mean() > self.y_nan_percent:
            self.y_nan_class = x[LABEL_COL].max() + 1

        X = x.drop(columns=[LABEL_COL]).copy()

        # categorical encoding
        for col in x.select_dtypes(include=['object']).columns:
            self.le[col] = LabelEncoder()
            X[col] = self.le[col].fit_transform(x[col].astype(str))

        # imputing
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        # variance threshold
        X = self.variance_thr.fit_transform(X)

        # normalization
        self.normalizer.fit(X)

    def transform(self, x):
        assert LABEL_COL in x.columns, f'label column "{LABEL_COL}" is not in dataframe, check dataframe format'

        # handle missing labels
        if self.y_nan_class is None:
            x = x[~x[LABEL_COL].isna()]
        else:
            x[LABEL_COL].fillna(self.y_nan_class, inplace=True)

        X = x.drop(columns=[LABEL_COL])
        y = x[LABEL_COL].astype(int)

        # categorical encoding
        for col in self.le.keys():
            X[col] = self.le[col].transform(X[col].astype(str))

        # features imputation
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)

        # variance threshold
        X = self.variance_thr.transform(X)

        # normalization
        X = self.normalizer.transform(X)

        res = pd.DataFrame(X, columns=self.variance_thr.get_feature_names_out(), index=x.index)
        res[LABEL_COL] = y

        return res
