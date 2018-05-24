import numpy as np


def calculate_contrast(y, y_type, slice_vector):
    if y_type == "numeric":
        contrast = _calculate_contrast_ks(y, slice_vector)
    else:
        contrast = _calculate_contrast_kld(y, slice_vector)
    return contrast


def _calculate_contrast_ks(y, slice_vector):
    y_cond = y[slice_vector]
    samples, N = len(y_cond), len(y)

    # TODO check
    current_div = 0
    size = min(50, int(0.5 * samples))
    cutpoints = np.random.choice(y_cond, size, False)
    for cut in cutpoints:
        sample_prob = np.sum(y_cond <= cut) / samples
        real_prob = np.sum(y <= cut) / N
        current_div = max(current_div, abs(sample_prob - real_prob))

    # TODO: replace with scipy?
    # from scipy.stats import ks_2samp
    # print("Scipy", ks_2samp(y, y[slice_vector]))
    return current_div


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
