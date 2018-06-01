import itertools
import numpy as np


def get_slices(X, types, n_select, n_iterations=100):
    # DISCUSS
    # TODO: create some more to have enough to select from
    n_vectors = int(np.ceil(n_iterations**(1 / len(types))))

    # TODO: slices should be int np.array (at least an array!)
    slice_pool = [None] * len(types)
    for i, col in enumerate(X):
        slice_pool[i] = {
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, n_select, n_vectors)

    slices = [None] * np.product([len(slices) for slices in slice_pool])
    for i, combination in enumerate(itertools.product(*slice_pool)):
        # TODO partial slicing should go here
        slices[i] = np.all(list(combination), axis=0)

    # TODO: remove emtpy slices
    if len(slices) > n_iterations:
        slices = np.asarray(slices)
        indices = np.random.choice(range(0, len(slices)), n_iterations)
        return slices[indices].tolist()
    return slices


def get_categorical_slices(X, n_select, n_vectors):
    # TODO: can be cached if categorical nans are extra category
    # TODO: caching: in deletion only set values 0 if samples are deleted
    # TODO: might change the size of the slice (store next value?)
    # TODO: sample from category to get more slices
    values, counts = np.unique(X, return_counts=True)
    value_dict = dict(zip(values, counts))

    slices = [None] * n_vectors
    for i in range(n_vectors):
        # TODO: tackle slice similarity here?
        values = np.random.permutation(values)

        selected_values, current_sum = [], 0
        for value in values:
            if current_sum >= n_select:
                break
            selected_values.append(value)
            current_sum += value_dict[value]
        slices[i] = np.isin(X, selected_values)
    return slices


def get_numerical_slices(X, n_select, n_vectors):
    # TODO: Think of how to support nan's here (probabilistic slicing)
    # TODO: adding salt?
    sorted_indices = np.argsort(X)

    max_start = X.shape[0] - n_select
    # TODO similarity using replace=False
    start_positions = np.random.choice(range(0, max_start), n_vectors)

    slices = [None] * n_vectors
    for i, start in enumerate(start_positions):
        start_value = X[sorted_indices[start]]
        end_value = X[sorted_indices[start + (n_select - 1)]]
        slices[i] = np.logical_and(X >= start_value, X <= end_value)
    return slices
