#cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np
from time import time

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

    cdef np.uint8_t[:,:] slices_int = cache["slices"].view(dtype=np.uint8, type=np.matrix)

    cdef double[:] contrasts = np.zeros(n)
    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            contrasts[i] = _calculate_max_dist(y_sorted, slices_int[i,:], lengths[i])
    return contrasts

cdef public double _calculate_max_dist(double[:] m, np.uint8_t[:] slice_, double n_c) nogil:
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

def _calculate_contrasts_kld(cache):
    values_m = cache["values"]
    probs_m = cache["probs"]
    sorted_y = cache["sorted"]
    slices = cache["slices"]

    cdfs = np.zeros((len(slices), len(values_m)))
    for i, s in enumerate(slices):
        cdfs[i, :] = _calculate_probs_kld(sorted_y[s], values_m)

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

def calculate_contrasts2(cache):
    return {
        "numeric": _calculate_contrasts_ks2,
        "nominal": _calculate_contrasts_kld2
    }[cache["type"]](cache)


def _calculate_contrasts_ks2(cache):
    cdef int n = len(cache["slices"]) 
    cdef int i = 0

    cdef double[:] y_sorted = cache["sorted"]
    cdef double[:] lengths = cache["lengths"]

    cdef np.float_t[:,:] slices_float = cache["slices"].view(dtype=np.float, type=np.matrix)

    cdef double[:] contrasts = np.zeros(n)
    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            contrasts[i] = _calculate_max_dist2(y_sorted, slices_float[i,:], lengths[i])
    return contrasts

cdef public double _calculate_max_dist2(double[:] m, np.float_t[:] slice_, double n_c) nogil:
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

def _calculate_contrasts_kld2(cache):
    values_m = cache["values"]
    probs_m = cache["probs"]
    sorted_y = cache["sorted"]
    slices = cache["slices"]

    cdfs = np.zeros((len(slices), len(values_m)))
    for i, s in enumerate(slices):
        cdfs[i, :] = _calculate_probs_kld2(sorted_y, s, values_m)

    cdfs += 1e-8
    return np.sum(cdfs * np.log2(cdfs / probs_m), axis=1)


def _calculate_probs_kld2(y, slice_, values_m):
    m = len(values_m)
    
    indices = np.searchsorted(y, values_m, side="right")
    weight_sum = np.sum(slice_)

    counts = np.zeros(m)
    prev = 0
    for i in range(m):
        counts[i] = np.sum(slice_[prev:indices[i]])
        prev = indices[i]
    
    return counts / weight_sum
