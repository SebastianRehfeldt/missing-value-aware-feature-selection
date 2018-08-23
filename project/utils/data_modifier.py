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
        X = _remove_with_mcar(data.X, missing_rate, features)
    elif missing_type == "predictive":
        X = _remove_by_class(data, missing_rate, features)
    elif missing_type == "NMAR":
        X = _remove_with_nmar(data, missing_rate, features)
    elif missing_type == "correlated":
        X = _remove_with_correlation(data, missing_rate, features)
    else:
        raise NotImplementedError

    for col in X:
        if data.f_types[col] == "nominal":
            X[col].fillna("?", inplace=True)
    return data.replace(X=X)


def _remove_with_mcar(X, missing_rate, features):
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

    if features is not None:
        X_orig[features] = new_X
        new_X = X_orig
    return new_X


def _remove_by_class(data, missing_rate, features):
    complete_X = data.X
    idx = np.where(data.y == mode(data.y).mode[0])[0]
    X = _remove_with_mcar(data.X.iloc[idx, :], missing_rate, features)
    complete_X.iloc[idx, :] = X
    return complete_X


def _remove_with_nmar(data, missing_rate, features):
    X = data.X
    if features is not None:
        X_orig = deepcopy(X)
        X = X[features]

    n_removals = round(X.shape[0] * missing_rate)
    sorted_indices = np.argsort(X, axis=0)
    for col in X.columns:
        # remove bottom or top values
        indices = sorted_indices[col]
        if np.random.choice(range(2), 1)[0] == 0:
            X[col].iloc[indices[:n_removals]] = np.nan
        else:
            X[col].iloc[indices[X.shape[0] - n_removals:]] = np.nan

    if features is not None:
        X_orig[features] = X
        X = X_orig
    return X


def _remove_with_correlation(data, missing_rate, features):
    X = data.X
    if features is not None:
        X_orig = deepcopy(X)
        X = X[features]

    n_removals = round(X.shape[0] * missing_rate)
    indices = np.random.choice(range(X.shape[0]), n_removals, False)
    dummy = np.zeros(X.shape[0], dtype=bool)
    dummy[indices] = True

    for col in X.columns:
        corr = np.random.normal(0, 0.5)
        corr = np.clip(corr, -1, 1)
        n_shuffle = int((1 - abs(corr)) * X.shape[0])
        indices = np.random.choice(range(X.shape[0]), n_shuffle, False)

        d = deepcopy(dummy) if corr >= 0 else deepcopy(~dummy)
        shuffled = d[indices]
        np.random.shuffle(shuffled)
        d[indices] = shuffled

        if corr < 0:
            n = d.sum()
            if n > n_removals:
                indices = np.random.choice(range(n), n - n_removals, False)
                indices = np.where(d)[0][indices]
                d[indices] = False
            else:
                indices = np.random.choice(
                    range(n_removals), n_removals - n, False)
                indices = np.where(~d)[0][indices]
                d[indices] = True

        X[col][d] = np.nan

    if features is not None:
        X_orig[features] = X
        X = X_orig
    return X
