import numpy as np


def calculate_contrasts(y_type, slices, cache):
    return {
        "numeric": _calculate_contrasts_ks,
        "nominal": _calculate_contrasts_kld
    }[y_type](slices, cache)


def _calculate_contrasts_ks(slices, cache):
    sorted_y = cache["sorted"]
    return [_calculate_contrast_ks(sorted_y, sorted_y[s]) for s in slices]


def _calculate_contrast_ks(y, y_cond):
    # As in implementation from scipy but without sorting first data
    n1 = y.shape[0]
    n2 = y_cond.shape[0]
    data_all = np.concatenate([y, y_cond])
    cdf1 = np.searchsorted(y, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(y_cond, data_all, side='right') / (1.0 * n2)
    return np.max(np.absolute(cdf1 - cdf2))


def _calculate_contrasts_kld(slices, cache):
    values_m = cache["values"]
    probs_m = cache["probs"]
    sorted_y = cache["sorted"]
    template = {value: 1e-8 for value in values_m}

    cdfs = np.zeros((len(slices), len(values_m)))
    for i, s in enumerate(slices):
        cdfs[i, :] = _calculate_probs_kld(sorted_y[s], template)
    return np.sum(cdfs * np.log2(cdfs / probs_m), axis=1)


def _calculate_probs_kld(y_cond, template):
    probs_c = template.copy()

    values_c, counts_c = np.unique(y_cond, return_counts=True)
    value_dict = dict(zip(values_c, counts_c / len(y_cond)))

    probs_c.update(value_dict)
    return list(probs_c.values())
