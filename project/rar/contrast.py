import numpy as np
from scipy.stats import ks_2samp


def calculate_contrast(y, y_cond, y_type):
    # TODO: allow missing values in target?
    return {
        "numeric": _calculate_contrast_ks,
        "nominal": _calculate_contrast_kld
    }[y_type](y, y_cond)


def _calculate_contrast_ks(y, y_cond):
    return ks_2samp(y, y_cond)[0]
    # TODO: replace with BP implementation?
    # TODO: check implementation
    """
    samples, N = len(y_cond), len(y)

    current_div = 0
    size = min(50, int(0.5 * samples))
    cutpoints = np.random.choice(y_cond, size, False)
    for cut in cutpoints:
        sample_prob = np.sum(y_cond <= cut) / samples
        real_prob = np.sum(y <= cut) / N
        current_div = max(current_div, abs(sample_prob - real_prob))
    return current_div
    """


def _calculate_contrast_kld(y, y_cond):
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
