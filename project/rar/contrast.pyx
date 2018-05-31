# cython: boundscheck=False, wraparound=False

from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def calculate_contrasts(y_type, slices, cache):
    return {
        "numeric": _calculate_contrasts_ks,
        "nominal": _calculate_contrasts_kld
    }[y_type](slices, cache)


def _calculate_contrasts_ks(slices, cache):
    cdef int n = len(slices) 
    cdef int i = 0
    cdef double[:] y_sorted = cache["sorted"]
    cdef bint[:,:] slices_int = np.asarray(slices, dtype=int)
    cdef int[:] slice_lengths = np.asarray([np.sum(s) for s in slices])

    cdef double[:] contrasts = np.zeros(len(slices))
    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            contrasts[i] = _calculate_contrast_ks(y_sorted, slices_int[i,:], slice_lengths[i])
    return contrasts

cdef public double _calculate_contrast_ks(double[:] m, bint[:] slice_, int n_c) nogil:
    if n_c == 0:
        return 0
    
    cdef int i = 0, j=0
    cdef int n_m = len(m)
    cdef int n = n_m + n_c
    cdef double *c = <double *> malloc(sizeof(double) * n_c)
    cdef double *y_all = <double *> malloc(sizeof(double) * n)

    cdef double counter_m = 0, counter_c = 0
    cdef double max_dist = 0, distance = 0
    for i in range(n_m - 1):
        counter_m += 1
        if slice_[i] == 1:
            counter_c += 1

        # calculate distance if value to the right is new value
        if m[i] != m[i+1]:
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
