#!python
#cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
import numpy as np

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
    for i in prange(n, schedule='static', nogil=True):
        contrasts[i] = _calculate_max_dist(y_sorted, slices_int[i,:], slice_lengths[i])
    return contrasts

cdef public double _calculate_max_dist(double[:] m, bint[:] slice_, int n_c) nogil:
    if n_c == 0:
        return 0
    
    cdef int i = 0
    cdef int n_m = len(m)

    cdef double m_step = 1.0 / n_m
    cdef double c_step = 1.0 / n_c

    cdef double counter_m = 0, counter_c = 0
    cdef double max_dist = 0, distance = 0
    for i in range(n_m - 1):
        counter_m += m_step
        if slice_[i]:
            counter_c += c_step

        # calculate distance if value to the right is new value
        if m[i] != m[i+1]:
            distance = counter_m - counter_c
            if (distance < 0):
                distance *= -1
            if distance > max_dist:
                max_dist = distance
    return max_dist

def _calculate_contrasts_kld(slices, cache):
    values_m = cache["values"]
    probs_m = cache["probs"]
    sorted_y = cache["sorted"]

    cdfs = np.zeros((len(slices), len(values_m)))
    for i, s in enumerate(slices):
        cdfs[i, :] = _calculate_probs_kld(sorted_y[s], values_m)

    # make sure that no division by 0 takes place
    cdfs += 1e-8
    return np.sum(cdfs * np.log2(cdfs / probs_m), axis=1)


def _calculate_probs_kld(y_cond, values_m):
    m = len(values_m)
    
    indices = np.searchsorted(y_cond, values_m, side="right") / len(y_cond)
    counts = np.zeros(m)

    prev = 0
    for i in range(m):
        counts[i] = indices[i] - prev
        prev = indices[i]
    return counts
