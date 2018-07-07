import numpy as np


def get_slices(X, types, **options):
    # collect slices for each column
    slices = [None] * X.shape[1]
    for i, col in enumerate(X):
        slices[i] = {
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, **options)
    slices = combine_slices(slices)

    # remove empty and very small slices
    return prune_slices(slices, options["min_samples"])


def combine_slices(slices):
    if len(slices) == 1:
        return slices[0]
    return np.multiply.reduce(slices, 0, dtype=slices[0].dtype)


# TODO: min_samples should differ between approaches
def prune_slices(slices, min_samples=3):
    sums = np.sum(slices, axis=1, dtype=float)
    indices = sums > min_samples
    if np.any(~indices):
        return slices[indices], sums[indices]
    return slices, sums


def get_numerical_slices(X, **options):
    n_iterations, n_select = options["n_iterations"], options["n_select"]

    if options["approach"] in ["partial", "fuzzy", "deletion"]:
        indices, nans = options["indices"], options["nans"]
    else:
        indices = np.argsort(X)

    # TODO: account for missing values (also increase max_start if range very small)
    max_start = X.shape[0] - n_select
    max_value = max(X)
    if options["approach"] in ["partial", "fuzzy", "deletion"]:
        # TODO: speed up?
        non_nan_count = indices.shape[0] - nans.sum()
        max_start = non_nan_count - n_select
        max_value = X[indices[non_nan_count - 1]]
        if max_start < 1:
            max_start = max(10, int(non_nan_count / 2))

    start_positions = np.random.choice(range(0, max_start), n_iterations)

    dtype = np.float16 if options["approach"] == "fuzzy" else bool
    slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
    for i, start in enumerate(start_positions):
        if options["should_sample"]:
            idx = indices[start:start + n_select]
            slices[i, idx] = True
        else:
            start_value = X[indices[start]]
            end_position = start + n_select
            end_value = min(X[indices[end_position]], max_value)
            slices[i] = np.logical_and(X >= start_value, X <= end_value)

    if options["approach"] == "partial":
        slices[:, nans] = True
    if options["approach"] == "fuzzy":
        slices[:, nans] = (n_select / non_nan_count) * options["weight"]
    return slices


def get_categorical_slices(X, **options):
    n_iterations, n_select = options["n_iterations"], options["n_select"]

    if options["approach"] in ["partial", "fuzzy", "deletion"]:
        values, counts = options["values"], options["counts"]
    else:
        values, counts = np.unique(X, return_counts=True)

    value_dict = dict(zip(values, counts))
    index_dict = {val: np.where(X == val)[0] for val in values}
    values_to_select = list(values)

    contains_nans = "?" in values_to_select
    if contains_nans:
        values_to_select.remove("?")

    dtype = np.float16 if options["approach"] == "fuzzy" else bool
    slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
    for i in range(n_iterations):
        values_to_select = np.random.permutation(values_to_select)
        current_sum = 0
        for value in values:
            current_sum += value_dict[value]

            if current_sum >= n_select:
                if options["should_sample"]:
                    n_missing = n_select - (current_sum - value_dict[value])
                    idx = np.random.choice(index_dict[value], n_missing, False)
                    slices[i, idx] = True
                else:
                    slices[i, index_dict[value]] = True
                break

            slices[i, index_dict[value]] = True

    if options["approach"] == "partial" and contains_nans:
        slices[:, index_dict["?"]] = True
    if options["approach"] == "fuzzy" and contains_nans:
        non_nan_count = X.shape[0] - value_dict["?"]
        w = (n_select / non_nan_count) * options["weight"]
        slices[:, index_dict["?"]] = w
    return slices
