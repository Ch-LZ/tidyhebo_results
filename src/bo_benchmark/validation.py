"""DataFrame boundary validation; index never determines observation pairing."""
import numpy as np
import pandas as pd


def validate_X(X, columns=None, n=None):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas.DataFrame')
    if X.empty or not X.columns.is_unique:
        raise ValueError('X must be nonempty with unique parameter columns')
    if columns is not None:
        if set(X.columns) != set(columns):
            raise ValueError(f'X columns must match {list(columns)}')
        X = X.loc[:, list(columns)]
    if n is not None and len(X) != n:
        raise ValueError(f'Expected exactly {n} candidate rows, got {len(X)}')
    return X.reset_index(drop=True).copy(deep=True)


def normalize_observations(X, y, columns=None):
    X = validate_X(X, columns)
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError('y must be a pandas.Series or numpy.ndarray')
    values = np.asarray(y)
    if values.shape not in ((len(X),), (len(X), 1)):
        raise ValueError('y must have shape (n,) or (n,1), matching X rows')
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('Scalar y must contain finite real numbers')
    return X, values.reshape(-1, 1).copy()
