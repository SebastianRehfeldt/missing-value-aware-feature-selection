#cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np

def calculate_contrasts(cache):
    return {
        "numeric": _calculate_contrasts_ks,
        "nominal": _calculate_contrasts_kld
    }[cache["type"]](cache)


def _calculate_contrasts_ks(cache):
    cdef int n = len(cache["slices"]) 
    cdef int i = 0

    cdef double[:] y_sorted = cache["sorted"]
    cdef double[:] lengths = cache["lengths"]

    # TODO: normalize slice sums to 1 which makes max dist calc faster
    cdef np.float_t[:,:] slices = cache["slices"].astype(np.float)

    cdef double[:] contrasts = np.zeros(n)
    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            contrasts[i] = _calculate_max_dist(y_sorted, slices[i,:], lengths[i])
    return contrasts

cdef public double _calculate_max_dist(double[:] m, np.float_t[:] slice_, double n_c) nogil:
    if n_c == 0:
        return 0
    
    cdef int i = 0, n_m = len(m)
    cdef double m_step = 1.0 / n_m

    cdef double counter_m = 0, counter_c = 0
    cdef double max_dist = 0, distance = 0
    for i in range(n_m - 1):
        counter_m += m_step
        counter_c += slice_[i]

        # calculate distance if value to the right is new value
        if m[i] != m[i+1]:
            distance = counter_m - (counter_c / n_c)
            if (distance < 0):
                distance *= -1
            if distance > max_dist:
                max_dist = distance
    return max_dist

def _calculate_contrasts_kld(cache):
    probs = cache["counts"]  / len(cache["sorted"])
    cdef int[:] counts = cache["counts"].astype(int)

    # TODO: normalize slice sums to 1 which makes max dist calc faster
    cdef np.float_t[:,:] slices = cache["slices"].astype(np.float)
    cdef double[:] lengths = cache["lengths"]

    cdef int i= 0, n = len(cache["slices"])
    cdef double[:,:] cdfs = np.zeros((len(slices), len(probs)))

    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            cdfs[i, :] = _calculate_probs_kld(counts, slices[i,:], lengths[i])

    return np.sum(cdfs * np.log2(cdfs / probs), axis=1)


cdef public double[:] _calculate_probs_kld(int[:] marginal_counts, np.float_t[:]slice_, double weight_sum) nogil:
    cdef int n_values = len(marginal_counts), i = 0
    cdef int n_samples = len(slice_)

    cdef double[:] conditional_counts
    with gil:
        conditional_counts = np.zeros(n_values)

    cdef int value_i = 0
    cdef int current_counts = 0
    for sample_i in range(n_samples):
        if current_counts == marginal_counts[value_i]:
            value_i += 1
            current_counts = 0

        conditional_counts[value_i] += slice_[sample_i]
        current_counts += 1

    for i in range(n_values):
        conditional_counts[i] /= weight_sum
        conditional_counts[i] += 0.0000001

    return conditional_counts
    