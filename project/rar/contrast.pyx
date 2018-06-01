#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
from time import time
import numpy as np

cdef extern from "math.h" nogil:
    double abs(double)

def calculate_contrasts(y_type, slices, cache):
    return {
        "numeric": _calculate_contrasts_ks,
        "nominal": _calculate_contrasts_kld
    }[y_type](slices, cache)


def _calculate_contrasts_ks(slices, cache):
    cdef int n = len(slices) 
    cdef int i = 0
    cdef double[:] y_sorted = cache["sorted"]

    # TODO: slices should be int np.array (at least an array!)
    slices = np.asarray(slices, dtype=int) 
    cdef bint[:,:] slices_int = slices
    cdef int[:] slice_lengths = np.sum(slices, axis=1)

    cdef double[:] contrasts = np.zeros(n)
    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            contrasts[i] = _calculate_contrast_ks(y_sorted, slices_int[i,:], slice_lengths[i])
    return contrasts

cdef public double _calculate_contrast_ks(double[:] m, bint[:] slice_, int n_c) nogil:
    if n_c == 0:
        return 0
    
    cdef int i = 0
    cdef int n_m = len(m)

    cdef double m_step = 1 / n_m
    cdef double c_step = 1 / n_c

    cdef double counter_m = 0, counter_c = 0
    cdef double max_dist = 0, distance = 0
    for i in range(n_m - 1):
        counter_m += m_step
        if slice_[i] == 1:
            counter_c += c_step

        # calculate distance if value to the right is new value
        if m[i] != m[i+1]:
            distance = abs(counter_m - counter_c)
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
