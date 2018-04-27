import numpy as np


def introduce_missing_values(data, missing_rate=0.25, missing_type="MCAR"):
    # Introduce missing values by a specified method
    total_samples = data.shape[0] * data.shape[1]
    n_removals = round(missing_rate * total_samples)

    incomplete_data = data.copy()
    if missing_type == "MCAR":
        return remove_with_mcar(incomplete_data, total_samples, n_removals)
    else:
        raise NotImplementedError


def remove_with_mcar(data, total_samples, n_removals):
    # Introduce missing values using MCAR
    mask_indices = np.random.choice(total_samples, n_removals, replace=False)
    mask = np.zeros(total_samples, dtype=bool)
    mask[mask_indices] = 1
    mask = np.reshape(mask, data.shape)

    data = data.where(mask == False)
    return data
