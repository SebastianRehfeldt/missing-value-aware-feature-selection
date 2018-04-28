"""
    Used to introduce missing values into artificial datasets.
    Could be used for real-world-datasets as well, but final missing rate will
    be different from specified one if missing values are already present.
"""
import numpy as np


def introduce_missing_values(data, feature_types, missing_rate=0.25, missing_type="MCAR"):
    # Introduce missing values by a specified method
    total_samples = data.shape[0] * data.shape[1]
    n_removals = round(missing_rate * total_samples)

    incomplete_data = data.copy()
    if missing_type == "MCAR":
        return remove_with_mcar(incomplete_data, feature_types, total_samples, n_removals)
    else:
        raise NotImplementedError


def remove_with_mcar(data, feature_types, total_samples, n_removals):
    # Introduce missing values using MCAR
    mask_indices = np.random.choice(total_samples, n_removals, replace=False)
    mask = np.zeros(total_samples, dtype=bool)
    mask[mask_indices] = 1
    mask = np.reshape(mask, data.shape)

    data = data.where(mask == False)
    for i in range(data.shape[1]):
        if feature_types[i] == "nominal":
            data.iloc[:, i].replace(np.nan, b"?", True)
    return data
