"""
    Used to introduce missing values into artificial datasets.
    Could be used for real-world-datasets as well, but final missing rate will
    be different from specified one if missing values are already present.
"""
import numpy as np


def introduce_missing_values(data, missing_rate=0.25, missing_type="MCAR"):
    """
    Introduce missing values by a specified method

    Arguments:
        data {Data} -- Data object where missing values should be inserted

    Keyword Arguments:
        missing_rate {float} -- Rate of how many missing values should be inserted (default: {0.25})
        missing_type {str} -- Approach for inserting missing values (default: {"MCAR"})
    """
    n_total_values = data.shape[0] * data.shape[1]
    n_removals = round(missing_rate * n_total_values)

    if missing_type == "MCAR":
        return _remove_with_mcar(data, n_total_values, n_removals)
    else:
        raise NotImplementedError


def _remove_with_mcar(data, n_total_values, n_removals):
    """
    Insert missing values completely at random 

    Arguments:
        data {Data} -- Data object where missing values should be inserted
        n_total_values {int} -- Number of total values in dataframe
        n_removals {int} -- Number of missing values being inserted
    """
    # Create mask where values should be inserted
    mask_indices = np.random.choice(n_total_values, n_removals, replace=False)
    mask = np.zeros(n_total_values, dtype=bool)
    mask[mask_indices] = 1
    mask = np.reshape(mask, data.shape)

    # Values in df with mask == True will be NaN
    features = data.features.where(mask == False)
    for col in data.features:
        if data.f_types[col] == "nominal":
            features[col].fillna(b"?", inplace=True)
    return data._replace(features=features)
