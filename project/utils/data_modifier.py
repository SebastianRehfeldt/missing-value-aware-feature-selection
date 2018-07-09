"""
    Used to introduce missing values into artificial datasets.
    Could be used for real-world-datasets as well, but final missing rate will
    be different from specified one if missing values are already present.
"""
import numpy as np


def introduce_missing_values(data,
                             missing_rate=0.25,
                             missing_type="MCAR",
                             seed=42):
    """
    Introduce missing values by a specified method

    Arguments:
        data {Data} -- Data object where missing values should be inserted

    Keyword Arguments:
        missing_rate {float} -- Rate of inserted nan's (default: {0.25})
        missing_type {str} -- Meachnism for missingness (default: {"MCAR"})
    """
    n_total_values = data.shape[0] * data.shape[1]
    n_removals = round(missing_rate * n_total_values)

    if missing_type == "MCAR":
        return _remove_with_mcar(data, n_total_values, n_removals, seed)
    else:
        raise NotImplementedError


def _remove_with_mcar(data, n_total_values, n_removals, seed):
    """
    Insert missing values completely at random

    Arguments:
        data {Data} -- Data object where missing values should be inserted
        n_total_values {int} -- Number of total values in dataframe
        n_removals {int} -- Number of missing values being inserted
    """
    # Create mask where values should be inserted
    np.random.seed(seed)
    rn = np.random.normal(size=data.shape)
    min_value = np.sort(rn, kind="mergesort", axis=None)[n_removals]
    mask = rn < min_value

    # Values in df with mask == True will be NaN
    features = data.X.where(mask == False)
    for col in data.X:
        if data.f_types[col] == "nominal":
            features[col].fillna("?", inplace=True)
    return data.replace(X=features)
