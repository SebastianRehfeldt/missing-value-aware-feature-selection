import itertools
import numpy as np


def get_slices(X, types, n_select, n_iterations=100):
    # DISCUSS
    # TODO: create some more to have enough to select from
    # TODO: calculate this before
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
        # TODO partial slicing should go here
        combinations = list(itertools.product(*pool))
        selected = np.random.choice(
            range(len(combinations)), n_iterations, False)

        slices = np.zeros((n_iterations, X.shape[0]), dtype=bool)
        for i, s in enumerate(selected):
            slices[i, :] = np.all(combinations[s], axis=0)
    else:
        slices = pool[0]

    # DISCUSS
    # remove empty and very small slices
    # TODO: pass to calculate_contrast in order to avoid recalculation
    sums = np.sum(slices, axis=1)
    indices = sums > 10
    if np.any(~indices):
        slices = slices[indices]

    # reduce to n_iterations and return
    if len(slices) > n_iterations:
        indices = np.random.choice(range(0, len(slices)), n_iterations)
        return slices[indices]
    return slices


def get_categorical_slices(X, n_select, n_vectors):
    # TODO: sample from category to get more slices
    values, counts = np.unique(X, return_counts=True)
    value_dict = dict(zip(values, counts))
    index_dict = {val: np.where(X == val)[0] for val in values}

    slices = np.zeros((n_vectors, X.shape[0]), dtype=bool)
    for i in range(n_vectors):
        # TODO: tackle slice similarity here?
        values = np.random.permutation(values)
        current_sum = 0
        for value in values:
            if current_sum >= n_select:
                break
            slices[i, index_dict[value]] = True
            current_sum += value_dict[value]
    return slices


def get_numerical_slices(X, n_select, n_vectors):
    # TODO: Think of how to support nan's here (probabilistic slicing)
    # TODO: similarity using replace=False
    sorted_indices = np.argsort(X)
    max_start = X.shape[0] - n_select
    start_positions = np.random.choice(range(0, max_start), n_vectors)

    slices = np.zeros((n_vectors, X.shape[0]), dtype=bool)
    for i, start in enumerate(start_positions):
        # DISCUSS
        """
        start_value = X[sorted_indices[start]]
        end_value = X[sorted_indices[start + (n_select - 1)]]
        slices[i] = np.logical_and(X >= start_value, X <= end_value)
        """
        idx = sorted_indices[start:start + n_select - 1]
        slices[i, idx] = True
    return slices
