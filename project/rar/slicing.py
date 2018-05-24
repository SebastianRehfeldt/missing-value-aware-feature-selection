import numpy as np


def get_slices(X, types, n):
    slice_vector = np.ones(X.shape[0], dtype=bool)
    for col in X:
        new_slice = {
            "nominal": get_categorical_slice,
            "numeric": get_numerical_slice
        }[types[col]](X[col].values, n)
        slice_vector = np.logical_and(slice_vector, new_slice)
    return slice_vector


def get_categorical_slice(X, n):
    # TODO: can be cached and pre_calculated
    # TODO: in deletion only set values 0 if samples are deleted
    # TODO: might change the size of the slice (store next value?)
    # TODO: tackle similarity here?
    # TODO: also add salt here?
    values = np.random.permutation(X)
    values, counts = np.unique(values, return_counts=True)

    selected_values, current_sum = [], 0
    for i, value in enumerate(values):
        if current_sum >= n:
            break
        selected_values.append(value)
        current_sum = counts[i]
    return np.isin(X, selected_values)


def get_numerical_slice(X, n):
    # TODO: Think of how to support nan's here (probabilic slicing)
    # TODO: get cached sorted indices from HICS
    # TODO: slice similarity by setting different randints here?
    # TODO: adding salt?
    max_start = X.shape[0] - n
    start = np.random.randint(0, max_start)
    end = start + (n - 1)

    sorted_indices = np.argsort(X)
    start_value = X[sorted_indices[start]]
    end_value = X[sorted_indices[end]]
    return np.logical_and(X >= start_value, X <= end_value)
