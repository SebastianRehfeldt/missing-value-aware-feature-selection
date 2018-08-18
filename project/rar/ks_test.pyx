#cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np

def calculate_ks_contrast(cache):
    cdef int i = 0,  n = len(cache["slices"]) 
    cdef double[:] y_sorted = cache["sorted"]

    slices = cache["slices"].astype(float).T
    slices /= np.sum(slices, axis=0)
    cdef np.float_t[:,:] slices_c = slices.T

    cdef double[:] contrasts = np.zeros(n)
    with nogil, parallel(num_threads=1):
        for i in prange(n, schedule='static'):
            contrasts[i] = _calculate_max_dist(y_sorted, slices_c[i,:])
    return contrasts

cdef public double _calculate_max_dist(double[:] m, np.float_t[:] slice_) nogil:
    cdef int i = 0, n_m = len(m)
    cdef double m_step = 1.0 / n_m

    cdef double counter_m = 0, counter_c = 0
    cdef double max_dist = 0, distance = 0
    for i in range(n_m - 1):
        counter_m += m_step
        counter_c += slice_[i]

        # calculate distance if value to the right is new value
        if m[i] != m[i+1]:
            distance = counter_m - counter_c
            if (distance < 0):
                distance *= -1
            if distance > max_dist:
                max_dist = distance
    return max_dist
