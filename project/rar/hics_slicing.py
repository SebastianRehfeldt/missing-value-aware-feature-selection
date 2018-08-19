import numpy as np
from scipy.stats import norm
from .hics_params import HICSParams


class HICSSlicing(HICSParams):
    def compute_slices(self, subspace, cache, X, **options):
        slices = [None] * len(subspace)
        X = self.data.X if X is None else X
        for i, col in enumerate(self.data.X[subspace]):
            x = X[col].values
            t = self.data.f_types[col]

            slices[i] = {
                "nominal": self.get_categorical_slices,
                "numeric": self.get_numerical_slices
            }[t](x, cache, col, **options)

        return self.combine_slices(slices)

    def combine_slices(self, slices):
        if len(slices) == 1:
            return slices[0]
        return np.multiply.reduce(slices, 0, dtype=slices[0].dtype)

    def prune_slices(self, slices, min_samples=3):
        sums = np.sum(slices, axis=1, dtype=float)
        indices = sums > min_samples
        if np.any(~indices):
            return slices[indices]
        return slices

    def get_numerical_slices(self, X, cache, col, **options):
        n_iterations = self.params["contrast_iterations"]
        n_select = options["n_select"]

        indices = cache["indices"] if cache is not None else np.argsort(X)
        nans = self.nans[col]

        nan_count = 0 if self.params["approach"] == "imputation" else np.sum(
            nans)
        mr = nan_count / X.shape[0]
        n_select = max(5, int(np.ceil(n_select * (1 - mr))))

        non_nan_count = X.shape[0] - nan_count
        if options["boost"]:
            max_start = min(10, non_nan_count - n_select)
            start_positions = list(np.arange(0, max_start, 2))
            min_start = max(non_nan_count - n_select - 10, 0)
            end_positions = list(
                np.arange(min_start, non_nan_count - n_select, 2))
            start_positions = start_positions + end_positions
            n_iterations = len(start_positions)
        else:
            max_start = non_nan_count - n_select
            start_positions = np.random.randint(0, max_start, n_iterations)

        dtype = np.float16 if self.params["approach"] == "fuzzy" else bool
        slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
        if self.params["weight_approach"] == "probabilistic":
            probs = np.zeros((n_iterations, nan_count))
        for i, start in enumerate(start_positions):
            end = min(start + n_select, non_nan_count - 1)
            idx = indices[start:end]
            if self.params["weight_approach"] == "new" and self.params["approach"] == "fuzzy":
                weight_nans = options["n_select"] - n_select
                #weight_nans = min(weight_nans, n_select)

                min_val, max_val = X[indices[start]], X[indices[end]]
                radius = min(0.25, max(0.05, (max_val - min_val) / 2))
                center_val = (min_val + max_val) / 2
                #center_val = np.median(X[indices[start]:indices[end]])

                n_dev = 0.01
                dev = 0.1
                X_complete = options["X_complete"][col].values[nans]
                noise = np.random.normal(0, n_dev, len(X_complete))
                #X_complete += np.clip(noise, -dev, dev)
                X_dist = np.abs(X_complete - center_val)

                try:
                    # TODO: distribute more equally and less weight to nans
                    a = 1
                    n = len(X_dist[X_dist <= radius])
                    a = 2 if weight_nans * 2 <= n else 1

                    # we need to select less than we have (we can pick the top or distribute equally)
                    if weight_nans * a <= n:
                        m = weight_nans
                        bla = np.argpartition(X_dist, m)[:m]
                        nan_values = np.zeros(nan_count)
                        nan_values[bla] = 1
                        slices[i, nans] = nan_values
                    else:
                        m = n
                        w = min(1, ((weight_nans - n) / (nan_count - n)))
                        slices[i, nans] = w

                        bla = np.argpartition(X_dist, m)[:m]
                        nan_values = np.zeros(nan_count)
                        nan_values[bla] = 1
                        slices[i, nans] = nan_values

                except:
                    print("EXCEPT")
                    print(m)
                    print(X_dist, len(X_dist))
                    print(max_val - min_val)
                    print(options["n_select"], n_select, nan_count)
                    print(1 / 0)

            slices[i, idx] = True
            if self.params["weight_approach"] == "probabilistic":
                min_val, max_val = X[indices[start]], X[indices[end]]
                if True:
                    count = ((X >= min_val) & (X <= max_val)).sum()
                    probs[i, :] = count / non_nan_count
                else:
                    prob = norm.cdf(max_val) - norm.cdf(min_val)
                    prob += norm.pdf(min_val)
                    probs[i, :] = prob

        if self.params["approach"] == "partial" and not options["boost"]:
            slices[:, nans] = True
        if self.params["approach"] == "fuzzy" and not self.params["weight_approach"] == "new":
            factor = self.params["weight"]**(1 / options["d"])
            if self.params["weight_approach"] == "probabilistic":
                slices[:, nans] = probs * factor
            else:
                slices[:, nans] = options["alpha"] * factor
        return slices

    def get_categorical_slices(self, X, cache, **options):
        n_iterations = self.params["contrast_iterations"]
        n_select = options["n_select"]

        if cache is not None:
            values, counts = cache["values"], cache["counts"]
        else:
            values, counts = np.unique(X, return_counts=True)

        value_dict = dict(zip(values, counts))
        index_dict = {val: np.where(X == val)[0] for val in values}

        nan_count = value_dict.get("?", 0)
        non_nan_count = X.shape[0] - nan_count
        mr = nan_count / X.shape[0]
        n_select = max(5, int(np.ceil(n_select * (1 - mr))))

        contains_nans = "?" in value_dict
        if contains_nans:
            index = np.where(values == "?")[0]
            values = np.delete(values, index)

        if options["boost"]:
            s_per_class = 3 if len(values) < 10 else 1
            n_iterations = len(values) * s_per_class

            if self.params["weight_approach"] == "probabilistic":
                probs = np.zeros((n_iterations, nan_count))

            dtype = np.float16 if self.params["approach"] == "fuzzy" else bool
            slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
            for i, value in enumerate(values):
                for j in range(s_per_class):
                    perm = np.random.permutation(value_dict[value])[:n_select]
                    slices[i * s_per_class + j, index_dict[value][perm]] = True
                    if self.params["weight_approach"] == "probabilistic":
                        probs[i, :] += value_dict[value] / non_nan_count

        else:
            if self.params["weight_approach"] == "probabilistic":
                probs = np.zeros((n_iterations, nan_count))
            dtype = np.float16 if self.params["approach"] == "fuzzy" else bool
            slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
            for i in range(n_iterations):
                values = np.random.permutation(values)
                current_sum = 0
                for value in values:
                    if self.params["weight_approach"] == "probabilistic":
                        probs[i, :] += value_dict[value] / non_nan_count
                    current_sum += value_dict[value]
                    if current_sum >= n_select:
                        n_missing = n_select - (
                            current_sum - value_dict[value])
                        perm = np.random.permutation(
                            value_dict[value])[:n_missing]
                        idx = index_dict[value][perm]
                        slices[i, idx] = True
                        break

                    slices[i, index_dict[value]] = True

        if self.params["approach"] == "partial" and contains_nans and not options["boost"]:
            slices[:, index_dict["?"]] = True
        if self.params["approach"] == "fuzzy" and contains_nans:
            factor = self.params["weight"]**(1 / options["d"])
            if self.params["weight_approach"] == "probabilistic":
                slices[:, index_dict["?"]] = probs * factor
            else:
                slices[:, index_dict["?"]] = options["alpha"] * factor
        return slices
