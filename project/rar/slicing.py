import numpy as np


def get_slices(X, types, n):
    slice_vector = np.ones(X.shape[0], dtype=bool)
    for col in X:
        if types[col] == "nominal":
            new_slice = get_categorical_slice(X[col].values, n)
        else:
            new_slice = get_numerical_slice(X[col].values, n)
        slice_vector = np.logical_and(slice_vector, new_slice)

    return slice_vector


def get_categorical_slice(X, n):
    values = np.random.permutation(X)

    selected_values = []
    current_sum = 0
    for value, count in np.unique(values, return_counts=True):
        if current_sum < n:
            selected_values.append(value)
            current_sum = count
        else:
            break

    return np.isin(X, selected_values)


def get_numerical_slice(X, n):
    max_start = X.shape[0] - n
    start = np.random.randint(0, max_start)
    end = start + (n - 1)

    sorted_indices = np.argsort(X)
    start_value = X[sorted_indices[start]]
    end_value = X[sorted_indices[end]]

    return np.logical_and(X >= start_value, X <= end_value)
