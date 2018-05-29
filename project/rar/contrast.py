import numpy as np


def calculate_contrast(y_cond, y_type, cache):
    # TODO: improve speed by batch calculation of kld?
    # TODO: check if kld or ks are > 1 (but no normalization for now)
    # TODO: different value ranges from tests?
    # use 1-exp(-KLD(P,Q)) to normalize kld
    return {
        "numeric": _calculate_contrast_ks,
        "nominal": _calculate_contrast_kld
    }[y_type](y_cond, cache)


def _calculate_contrast_ks(y_cond, cache):
    # As in implementation from scipy but without sorting first data
    # TODO: check how different dimensions are handled
    y = cache["sorted"]
    y_cond = np.sort(y_cond)
    n1 = y.shape[0]
    n2 = y_cond.shape[0]
    data_all = np.concatenate([y, y_cond])
    cdf1 = np.searchsorted(y, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(y_cond, data_all, side='right') / (1.0 * n2)
    return np.max(np.absolute(cdf1 - cdf2))


def _calculate_contrast_kld(y_cond, cache):
    values_m = cache["values"]
    probs_m = cache["probs"]
    probs_c = {value: 1e-8 for value in values_m}

    values_c, counts_c = np.unique(y_cond, return_counts=True)
    value_dict = dict(zip(values_c, counts_c / len(y_cond)))

    probs_c.update(value_dict)
    probs_c = list(probs_c.values())
    return np.sum(probs_c * np.log2(probs_c / probs_m))
