import numpy as np
from libc.math cimport isnan, sqrt

def custom_distance(x1, x2, **kwargs):
    cdef str[:] f_types = kwargs.get("f_types")
    cdef int nominal_distance = kwargs.get("nominal_distance", 1)

    cdef double squared_dist = 0
    cdef int n_complete = 0
    cdef int n = len(x1)
    cdef double dist = 0
    # TODO check that f_type matches the right feature
    # print(x1, self.data.f_types)
    x1 = assert_series(x1)
    x2 = assert_series(x2)
    assert (len(x1) == len(x2)), "samples have different lengths"
    for i in range(n):
        is_numerical = f_types[i] == "numeric"

        # Only sum up distances between complete pairs
        if is_numerical and not isnan(x1[i]) and not isnan(x2[i]):
            n_complete += 1
            dist = x1[i] - x2[i]
            squared_dist += dist**2

        if not is_numerical:
            n_complete += 1
            if not x1[i] == x2[i]:
                squared_dist += nominal_distance

    return sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf