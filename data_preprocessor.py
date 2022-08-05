import time

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PowerTransformer, LabelEncoder, OrdinalEncoder, StandardScaler
from data_formatting import LABEL_COL


class DataPreprocessor:
    def __init__(self, y_nan_percent=0.1):
        """
        :param y_nan_percent: float, if the percentage of NaNs in y is more than that, we consider the NaN as an
                              additional class
        """
        self.numeric_transformer = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy="mean"))
        ])

        # impute and encode dummy variables for categorical data
        self.categorical_transformer = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy="constant")),
            ('encode', OrdinalEncoder())
        ])

        self.data_transformer = None
        self.le = LabelEncoder()

        self.variance_thr = VarianceThreshold()
        self.normalizer = PowerTransformer()

        self.y_nan_percent = y_nan_percent
        self.ignore_nan_labels = False

    def fit(self, X, y):
        # handle missing labels
        if y.isna().mean() > self.y_nan_percent:
            self.ignore_nan_labels = True

        self.le.fit(y)

        self.data_transformer = make_column_transformer(
            (self.numeric_transformer, X.select_dtypes(exclude=['object']).columns),
            (self.categorical_transformer, X.select_dtypes(include=['object']).columns))

        X = pd.DataFrame(self.data_transformer.fit_transform(X.copy()), columns=X.columns)

        # variance threshold
        X = self.variance_thr.fit_transform(X)

        # normalization
        self.normalizer.fit(X)

    def transform(self, X_org, y_org):
        X = X_org.copy()
        y = y_org.copy()

        # handle missing labels
        if self.ignore_nan_labels:
            X, y = X[~y.isna()], y[~y.isna()]
        y = self.le.transform(y)

        X_index = X.index

        # data transforming
        X = pd.DataFrame(self.data_transformer.transform(X), columns=X.columns)

        # variance threshold
        X = self.variance_thr.transform(X)

        # normalization
        X = self.normalizer.transform(X)

        X = pd.DataFrame(X, columns=self.variance_thr.get_feature_names_out(), index=X_index)

        return X, y
