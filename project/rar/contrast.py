import numpy as np
from .ks_test import calculate_ks_contrast


def calculate_contrasts(cache):
    return {
        "numeric": calculate_ks_contrast,
        "nominal": _calculate_kld_contrast
    }[cache["type"]](cache)


def _calculate_kld_contrast(cache):
    counts = cache["counts"]
    slices = cache["slices"]

    probs = counts / len(cache["sorted"])
    cdfs = np.zeros((len(slices), len(probs)))

    p = 0
    for i, count in enumerate(counts):
        cdfs[:, i] = np.sum(slices[:, p:p + count], axis=1)
        p += count

    cdfs += 1e-8
    cdfs /= np.sum(cdfs, axis=1)[:, None]
    return np.sum(cdfs * np.log2(cdfs / probs), axis=1)
