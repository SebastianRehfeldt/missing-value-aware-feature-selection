import numpy as np
from scipy.stats import ks_2samp


def calculate_contrast(y, y_type, slice_vector):
    if y_type == "numeric":
        contrast = _calculate_contrast_kld(y, slice_vector)
    else:
        contrast = _calculate_contrast_ks(y, slice_vector)

    return contrast


def _calculate_contrast_kld(y, slice_vector):
    y_cond = y[slice_vector]
    values_m, counts_m = np.unique(y, return_counts=True)
    values_c, counts_c = np.unique(y_cond, return_counts=True)

    probs_m = counts_m / len(y)
    probs_c = {}
    for value in values_m:
        probs_c[value] = 0.0000001

    for i, value in enumerate(values_c):
        probs_c[value] = counts_c[i] / len(y_cond)
    probs_c = list(probs_c.values())

    return (probs_c * np.log2(probs_c / probs_m)).sum()


def _calculate_contrast_ks(y, slice_vector):
    # TODO: return ks statistic or p_value?
    return ks_2samp(y, y[slice_vector])[0]