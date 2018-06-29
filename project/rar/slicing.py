import numpy as np


def get_slices(X, types, n_select, n_iterations):
    # collect slices for each column
    slices = [None] * X.shape[1]
    for i, col in enumerate(X):
        slices[i] = {
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, n_select, int(1.25 * n_iterations))
    slices = combine_slices(slices)

    # remove empty and very small slices
    return prune_slices(slices, n_iterations)


def get_partial_slices(X, indices, nans, f_type, n_select, n_iterations,
                       approach, should_sample):
    # TODO: REMOVE THIS AND COMBINE WITH GET SLICES
    if f_type == "numeric":
        slices = get_slices_num(X, indices, nans, n_select, n_iterations,
                                approach, should_sample)
    else:
        slices = get_categorical_slices(X, n_select, n_iterations, approach,
                                        should_sample)
    return slices


def combine_slices(slices):
    dimension = len(slices)
    if dimension == 1:
        return slices[0]
    elif dimension == 2:
        return np.logical_and(slices[0], slices[1])
    else:
        slices = np.logical_and(slices[0], slices[1])
        for i in range(2, dimension):
            slices.__iand__(slices[i])
    return slices


def prune_slices(slices, n_iterations, min_samples=3):
    # TODO: make min samples a param and test (should adapt to mr)
    sums = np.sum(slices, axis=1)
    indices = sums > min_samples
    if np.any(~indices):
        slices = slices[indices]
        sums = sums[indices]

    if len(slices) > n_iterations:
        indices = np.random.choice(range(0, len(slices)), n_iterations)
        return slices[indices], sums[indices]
    return slices, sums


def get_numerical_slices(X, n_select, n_vectors):
    # TODO: COMBINE WITH THE ONE BELOW
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


def get_slices_num(X,
                   indices,
                   nans,
                   n_select,
                   n_iterations,
                   approach="partial",
                   should_sample=False):

    # TODO: sort indices for deletion approach
    # TODO: ACCOUNT FOR MISSING VALUES (also increase max_start if very small)
    # TODO: reorder function arguments
    max_start = X.shape[0] - n_select
    if approach == "partial":
        non_nan_count = indices.shape[0] - nans.sum()
        max_start = non_nan_count - n_select
        if max_start < 1:
            max_start = max(10, int(non_nan_count / 2))

    start_positions = np.random.choice(range(0, max_start), n_iterations)
    slices = np.zeros((n_iterations, X.shape[0]), dtype=bool)
    for i, start in enumerate(start_positions):
        if should_sample:
            idx = indices[start:start + n_select - 1]
            slices[i, idx] = True
        else:
            # TODO: improve speed
            start_value = X[indices[start]]
            end_position = start + (n_select - 1)
            end_position = min(end_position, non_nan_count - 1)
            end_value = X[indices[end_position]]
            slices[i] = np.logical_and(X >= start_value, X <= end_value)

    if approach == "partial":
        slices[:, nans] = True
    return slices


def get_categorical_slices(X,
                           n_select,
                           n_iterations,
                           approach="deletion",
                           should_sample=False):
    # TODO: REFACTOR PARAMS
    # TODO: cache values and counts in partial approach
    values, counts = np.unique(X, return_counts=True)
    value_dict = dict(zip(values, counts))
    index_dict = {val: np.where(X == val)[0] for val in values}
    values_to_select = list(values)

    if approach == "partial":
        contains_nans = "?" in values_to_select
        if contains_nans:
            values_to_select.remove("?")

    slices = np.zeros((n_iterations, X.shape[0]), dtype=bool)
    for i in range(n_iterations):
        values_to_select = np.random.permutation(values_to_select)
        current_sum = 0
        for value in values:
            current_sum += value_dict[value]

            if current_sum >= n_select:
                if should_sample:
                    n_missing = n_select - (current_sum - value_dict[value])
                    idx = np.random.choice(index_dict[value], n_missing, False)
                    slices[i, idx] = True
                else:
                    slices[i, index_dict[value]] = True
                break

            slices[i, index_dict[value]] = True

    if approach == "partial" and contains_nans:
        slices[:, index_dict["?"]] = True
    return slices
