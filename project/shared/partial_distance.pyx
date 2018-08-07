#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from libc.math cimport isnan, sqrt

def partial_distance_orange(x1, x2, **kwargs):
    cdef str[:] f_types = kwargs["f_types"]
    cdef int nominal_distance = kwargs["nominal_distance"]

    cdef double squared_dist = 0
    cdef int n = len(x1), i = 0
    cdef double dist = 0

    for i in range(n):
        is_numerical = f_types[i] == "numeric"

        if is_numerical:
            # both missing
            if isnan(x1[i]) and isnan(x2[i]):
                squared_dist += 1
            # 1 missing (squared of known + half std)
            elif isnan(x1[i]):
                dist = x2[i]**2
                squared_dist += (dist + 0.5)
            elif isnan(x2[i]):
                dist = x1[i]**2
                squared_dist += (dist + 0.5)
            # euclidean dist if both present
            else:
                dist = x1[i] - x2[i]
                squared_dist += dist**2
        else:
            # both missing
            if (x1[i] == "?") and (x2[i] == "?"):
                squared_dist += nominal_distance
                continue
                
            # 1 missing (1 - prob of present value)
            prob = 0.5
            if x1[i] == "?":
                squared_dist += (1 - prob)
            if x2[i] == "?":
                squared_dist += (1 - prob)

            # add distance if they do not match
            if not x1[i] == x2[i]:
                squared_dist += nominal_distance
    return sqrt(squared_dist)

def partial_distance(x1, x2, **kwargs):
    cdef str[:] f_types = kwargs["f_types"]
    cdef int nominal_distance = kwargs["nominal_distance"]

    cdef double squared_dist = 0
    cdef int n_complete = 0
    cdef int n = len(x1), i = 0
    cdef double dist = 0

    for i in range(n):
        is_numerical = f_types[i] == "numeric"

        # Only sum up distances between complete pairs
        if is_numerical:
            if not isnan(x1[i]) and not isnan(x2[i]):
                n_complete += 1
                dist = x1[i] - x2[i]
                squared_dist += dist**2
        else:
            if (not x1[i] == "?") and (not x2[i] == "?"):
                n_complete += 1
                if not x1[i] == x2[i]:
                    squared_dist += nominal_distance
    return sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf