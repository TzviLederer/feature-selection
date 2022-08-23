import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder


def build_data_preprocessor(X, memory=None):
    numeric_transformer = SimpleImputer(missing_values=np.nan, strategy="mean")
    categorical_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="constant"), OrdinalEncoder())
    column_transformer = make_column_transformer(
        (numeric_transformer, X.select_dtypes(include=['number']).columns),
        (categorical_transformer, X.select_dtypes(exclude=['number']).columns),
        verbose_feature_names_out=False)

    return make_pipeline(column_transformer, VarianceThreshold(), PowerTransformer(), memory=memory)


class DataPreprocessorWrapper(BaseEstimator):
    def __init__(self, estimator):
        """
        Needed because imblearn do not excepts sklearn pipelines inside its own pipeline
        """

        self.estimator = estimator
        self.feature_names_in_ = None

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        self.feature_names_in_ = self.estimator.feature_names_in_
        return self

    def transform(self, X, y=None, **kwargs):
        return self.estimator.transform(X, **kwargs)

    def get_feature_names_out(self, **kwargs):
        return self.estimator.get_feature_names_out(**kwargs)