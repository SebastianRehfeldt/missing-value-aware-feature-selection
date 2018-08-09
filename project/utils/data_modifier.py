"""
    Used to introduce missing values into artificial datasets.
    Could be used for real-world-datasets as well, but final missing rate will
    be different from specified one if missing values are already present.
"""
import numpy as np
from copy import deepcopy
from scipy.stats import mode


def introduce_missing_values(data,
                             missing_rate=0.25,
                             missing_type="MCAR",
                             seed=42,
                             features=None):
    np.random.seed(seed)

    if missing_type == "MCAR":
        X = _remove_with_mcar(data.X, data.f_types, missing_rate, features)
        return data.replace(X=X)
    elif missing_type == "predictive":
        return _remove_by_class(data, missing_rate, features)
    else:
        raise NotImplementedError


def _remove_with_mcar(X, f_types, missing_rate, features):
    # Create mask where values should be inserted
    if features is not None:
        X_orig = deepcopy(X)
        X = X[features]

    rn = np.random.normal(size=X.shape)
    n_removals = round(missing_rate * X.shape[0] * X.shape[1])
    min_value = np.sort(rn, kind="mergesort", axis=None)[n_removals]
    mask = rn < min_value

    # Values in df with mask == True will be NaN
    new_X = X.where(mask == False)
    for col in X:
        if f_types[col] == "nominal":
            new_X[col].fillna("?", inplace=True)

    if features is not None:
        X_orig[features] = new_X
        new_X = X_orig
    return new_X


def _remove_by_class(data, missing_rate, features):
    complete_X = data.X
    idx = np.where(data.y == mode(data.y).mode[0])[0]
    X = _remove_with_mcar(data.X.iloc[idx, :], data.f_types, missing_rate,
                          features)
    complete_X.iloc[idx, :] = X
    return data.replace(X=complete_X)
