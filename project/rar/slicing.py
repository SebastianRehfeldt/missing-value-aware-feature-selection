import numpy as np


def get_slices(X, types, cache, **options):
    slices = [None] * X.shape[1]
    for i, col in enumerate(X):
        slices[i] = {
            "nominal": get_categorical_slices,
            "numeric": get_numerical_slices
        }[types[col]](X[col].values, cache, **options)

    return combine_slices(slices)


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


def get_numerical_slices(X, cache, **options):
    n_iterations, n_select = options["n_iterations"], options["n_select"]

    indices = cache["indices"] if cache is not None else np.argsort(X)
    nans = cache["nans"] if cache is not None else np.isnan(X)

    nan_count = 0 if options["approach"] == "imputation" else np.sum(nans)
    mr = nan_count / X.shape[0]
    n_select = max(5, int(np.ceil(n_select * (1 - mr))))

    non_nan_count = X.shape[0] - nan_count
    max_start = non_nan_count - n_select
    start_positions = np.random.randint(0, max_start, n_iterations)

    dtype = np.float16 if options["approach"] == "fuzzy" else bool
    slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
    for i, start in enumerate(start_positions):
        end = min(start + n_select, non_nan_count - 1)
        idx = indices[start:end]
        slices[i, idx] = True

    if options["approach"] == "partial":
        slices[:, nans] = True
    if options["approach"] == "fuzzy":
        factor = options["weight"]**(1 / options["d"])
        slices[:, nans] = options["alpha"] * factor
    return slices


def get_categorical_slices(X, cache, **options):
    n_iterations, n_select = options["n_iterations"], options["n_select"]

    if cache is not None:
        values, counts = cache["values"], cache["counts"]
    else:
        values, counts = np.unique(X, return_counts=True)

    value_dict = dict(zip(values, counts))
    index_dict = {val: np.where(X == val)[0] for val in values}

    nan_count = value_dict.get("?", 0)
    mr = nan_count / X.shape[0]
    n_select = max(5, int(np.ceil(n_select * (1 - mr))))

    contains_nans = "?" in value_dict
    if contains_nans:
        index = np.where(values == "?")[0]
        values = np.delete(values, index)

    dtype = np.float16 if options["approach"] == "fuzzy" else bool
    slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
    for i in range(n_iterations):
        values = np.random.permutation(values)
        current_sum = 0
        for value in values:
            current_sum += value_dict[value]
            if current_sum >= n_select:
                n_missing = n_select - (current_sum - value_dict[value])
                perm = np.random.permutation(value_dict[value])[:n_missing]
                idx = index_dict[value][perm]
                slices[i, idx] = True
                break

            slices[i, index_dict[value]] = True

    if options["approach"] == "partial" and contains_nans:
        slices[:, index_dict["?"]] = True
    if options["approach"] == "fuzzy" and contains_nans:
        non_nan_count = X.shape[0] - value_dict["?"]
        factor = options["weight"]**(1 / options["d"])
        w = (n_select / non_nan_count) * factor
        slices[:, index_dict["?"]] = w
    return slices
