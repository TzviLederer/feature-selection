import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder


def build_data_preprocessor(X, memory=None):
    return make_pipeline(build_column_transformer(X), VarianceThreshold(), PowerTransformer(), memory=memory)


def build_column_transformer(X):
    numeric_transformer = SimpleImputer(missing_values=np.nan, strategy="mean")
    categorical_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy="constant"), OrdinalEncoder())

    return make_column_transformer(
        (numeric_transformer, X.select_dtypes(include=['number']).columns),
        (categorical_transformer, X.select_dtypes(exclude=['number']).columns),
        verbose_feature_names_out=False)