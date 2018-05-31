import numpy as np
cimport cython

def calculate_contrasts(y_type, slices, cache):
    return {
        "numeric": _calculate_contrasts_ks,
        "nominal": _calculate_contrasts_kld
    }[y_type](slices, cache)


def _calculate_contrasts_ks(slices, cache):
    sorted_y = cache["sorted"]
    return [_calculate_contrast_ks(sorted_y, sorted_y[s]) for s in slices]

@cython.boundscheck(False)
@cython.wraparound(False)
def _calculate_contrast_ks(y, y_cond):
    # As in implementation from scipy but without sorting first data
    cdef int n_m = y.shape[0]
    cdef int n_c = y_cond.shape[0]

    if n_c == 0:
        return 0

    cdef int n = n_m + n_c
    cdef double[:] data_all = np.concatenate([y, y_cond])
    
    cdef double[:] m = y 
    cdef double[:] c = y_cond 
    cdef int counter_m = 0
    cdef int counter_c = 0

    cdef double max_dist = 0
    cdef double distance = 0
    cdef double current_value = 0

    with nogil:
        for i in range(n_m + n_c):
            current_value = data_all[i]
            while counter_m < n_m and m[counter_m] <= current_value:
                counter_m += 1

            while counter_c < n_c and c[counter_c] <= current_value:
                counter_c += 1

            distance = counter_m / n_m - counter_c / n_c
            if distance < 0:
                distance *= -1

            if distance > max_dist:
                max_dist = distance
    return max_dist


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
