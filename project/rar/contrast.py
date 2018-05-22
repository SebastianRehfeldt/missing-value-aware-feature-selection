import numpy as np


def calculate_contrast(y, y_type, slice_vector):
    dist_marginal = np.unique(y, return_counts=True)
    dist_cond = np.unique(y[slice_vector], return_counts=True)

    return 1
