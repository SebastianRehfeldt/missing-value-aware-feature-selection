import itertools
import numpy as np


def get_slices(X, types, n_select, n_iterations, slicing_method):
    slices = {
        "mating": get_slices_by_mating,
        "simple": get_slices_simple,
    }[slicing_method](X, types, n_select, int(1.25 * n_iterations))

    # remove empty and very small slices
    sums = np.sum(slices, axis=1)
    indices = sums > 10
    if np.any(~indices):
        slices = slices[indices]
        sums = sums[indices]

    # reduce to n_iterations and return
    if len(slices) > n_iterations:
        indices = np.random.choice(range(0, len(slices)), n_iterations)
        return slices[indices], sums[indices]
    return slices, sums


def get_slices_simple(X, types, n_select, n_iterations):
    slices = np.ones((n_iterations, X.shape[0]), dtype=bool)
    for col in X:
        slices.__iand__({
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, n_select, n_iterations))
    return slices


def get_slices_by_mating(X, types, n_select, n_iterations):
    n_vectors = int(np.ceil(n_iterations**(1 / len(types))))

    # pooling
    pool = [None] * len(types)
    for i, col in enumerate(X):
        pool[i] = {
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, n_select, n_vectors)

    # mating
    if len(types) > 1:
        combinations = list(itertools.product(*pool))
        selected = np.random.choice(
            range(len(combinations)), n_iterations, False)

        slices = np.zeros((n_iterations, X.shape[0]), dtype=bool)
        for i, s in enumerate(selected):
            slices[i, :] = np.all(combinations[s], axis=0)
    else:
        slices = pool[0]
    return slices


def get_categorical_slices(X, n_select, n_vectors):
    values, counts = np.unique(X, return_counts=True)
    value_dict = dict(zip(values, counts))
    index_dict = {val: np.where(X == val)[0] for val in values}

    slices = np.zeros((n_vectors, X.shape[0]), dtype=bool)
    for i in range(n_vectors):
        values = np.random.permutation(values)
        current_sum = 0
        for value in values:
            if current_sum >= n_select:
                break
            slices[i, index_dict[value]] = True
            current_sum += value_dict[value]
    return slices


def get_numerical_slices(X, n_select, n_vectors):
    sorted_indices = np.argsort(X)
    max_start = X.shape[0] - n_select
    start_positions = np.random.choice(range(0, max_start), n_vectors)

    slices = np.zeros((n_vectors, X.shape[0]), dtype=bool)
    for i, start in enumerate(start_positions):
        start_value = X[sorted_indices[start]]
        end_value = X[sorted_indices[start + (n_select - 1)]]
        slices[i] = np.logical_and(X >= start_value, X <= end_value)
        """
        idx = sorted_indices[start:start + n_select - 1]
        slices[i, idx] = True
        """
    return slices


def get_slices_num(X, indices, nans, n_select, n_iterations):
    slices = np.zeros((n_iterations, X.shape[0]), dtype=bool)

    non_nan_count = indices.shape[0] - nans.sum()
    max_start = non_nan_count - n_select
    if max_start < 1:
        # TODO: ACCOUNT FOR MISSING VALUES (also increase max_start if very small)
        # print("No starting positions for slice")
        max_start = max(10, int(non_nan_count / 2))

    start_positions = np.random.choice(range(0, max_start), n_iterations)
    for i, start in enumerate(start_positions):
        """
        start_value = X[indices[start]]
        end_position = start + (n_select - 1)
        end_position = min(end_position, non_nan_count - 1)
        end_value = X[indices[end_position]]
        # TODO: improve speed here
        slices[i] = np.logical_and(X >= start_value, X <= end_value)
        """
        idx = indices[start:start + n_select - 1]
        slices[i, idx] = True
    return slices


def get_partial_slices(X, indices, nans, f_type, n_select, n_iterations):
    if f_type == "numeric":
        slices = get_slices_num(X, indices, nans, n_select, n_iterations)
        slices[:, nans] = True
    return slices
