from typing import Tuple
import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_exporter
def export_data(
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], *args, **kwargs
) -> Tuple[
    csr_matrix,
    csr_matrix,
    csr_matrix,
    pd.Series,
    pd.Series,
    pd.Series,
    BaseEstimator,
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    X, _, _ = vectorize_features(select_features(df))
    y: pd.Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train),
        select_features(df_val),
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv

@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    *args,
) -> None:
    assert(
        X.shape[0] == 105870
    ), f'Entire dataset should have 105870 examples, but has {X.shape[0]}'
    assert(
        X.shape[1] == 7027
    ), f'Entire dataset should have 7027 features, but has {X.shape[0]}'
    assert(
        len(y.index) == X.shape[0]
    ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'