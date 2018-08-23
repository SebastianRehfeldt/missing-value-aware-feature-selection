import numpy as np
from scipy.stats import norm
from .hics_params import HICSParams


class HICSSlicing(HICSParams):
    def get_slices(self, subspace, n=None, d=None, boost=False, X=None):
        d = d or len(subspace)
        options = {
            "boost": boost,
            "d": d,
            "alpha": self.alphas_d[d],
            "n_select": n or self.n_select_d[d],
        }
        return self.compute_slices(subspace, X, **options)

    def compute_slices(self, subspace, X, **options):
        slices = [None] * len(subspace)
        X = self.data.X if X is None else X
        for i, col in enumerate(self.data.X[subspace]):
            x = X[col].values
            t = self.data.f_types[col]
            f_cache = self.feature_cache.get(col)

            slices[i] = {
                "nominal": self.get_categorical_slices,
                "numeric": self.get_numerical_slices
            }[t](x, f_cache, col, **options)

        return self.combine_slices(slices)

    def combine_slices(self, slices):
        if len(slices) == 1:
            return slices[0]
        return np.multiply.reduce(slices, 0, dtype=slices[0].dtype)

    def prune_slices(self, slices, min_samples=None):
        min_samples = min_samples or self.params["min_samples"]
        sums = np.sum(slices, axis=1, dtype=float)
        indices = sums > min_samples
        if np.any(~indices):
            return slices[indices]
        return slices

    def get_params(self, col, options):
        nans = self.nans[col]
        mr = self.missing_rates[col]
        nan_count = self.nan_sums[col]
        non_nan_count = self.data.shape[0] - nan_count
        n_select = max(5, int(np.ceil(options["n_select"] * (1 - mr))))
        return nans, nan_count, non_nan_count, n_select

    def get_start_pos(self, non_nan_count, n_select, boost):
        n_iterations = self.params["contrast_iterations"]
        if boost:
            max_start = min(10, non_nan_count - n_select)
            start_positions = list(np.arange(0, max_start, 2))
            min_start = max(non_nan_count - n_select - 10, 0)
            end_positions = list(
                np.arange(min_start, non_nan_count - n_select, 2))
            start_positions = start_positions + end_positions
        else:
            max_start = non_nan_count - n_select
            start_positions = np.random.randint(0, max_start, n_iterations)
        return start_positions

    def get_X_dist(self, col, center, nans):
        n_dev = 0.01
        dev = 0.1
        X_complete = self.X_complete[col].values[nans]
        noise = np.random.normal(0, n_dev, len(X_complete))
        #X_complete += np.clip(noise, -dev, dev)
        return np.abs(X_complete - center)

    def get_weight(self, X_dist, weight_nans, radius, nan_count):
        if self.params["dist_method"] == "distance" and len(X_dist) > 0:
            weights = (1 - (X_dist / np.max(X_dist)))
            weights = (weights / np.sum(weights)) * weight_nans
            return np.clip(weights, 0, 1)

        a = 1
        n = len(X_dist[X_dist <= radius])
        a = 2 if weight_nans * 2 <= n else 1

        nan_values = np.zeros(nan_count)
        if weight_nans * a <= n:
            m = weight_nans
            closest = np.argpartition(X_dist, m)[:m]
        else:
            w = min(1, ((weight_nans - n) / (nan_count - n)))
            nan_values[:] = w
            closest = np.argpartition(X_dist, n)[:n]
        nan_values[closest] = 1
        return nan_values

    def get_probs(self, min_val, max_val):
        prob = 0
        if self.params["weight_approach"] == "proba":
            prob = norm.cdf(max_val) - norm.cdf(min_val)
            prob += norm.pdf(min_val)
        return prob

    def update_nans(self, options, probs, weights):
        if options["boost"]:
            return 0

        if self.params["approach"] == "partial":
            return True

        if self.params["approach"] == "fuzzy":
            factor = self.params["weight"]**(1 / options["d"])
            w = {
                "imputed": weights,
                "alpha": options["alpha"],
                "proba": probs,
            }.get(self.params["weight_approach"], options["alpha"])
            return w * factor

    def get_numerical_slices(self, X, cache, col, **options):
        # PREPARATION
        is_fuzzy = self.params["approach"] == "fuzzy"
        indices = cache["indices"] if cache is not None else np.argsort(X)
        nans, nan_sum, non_nan_sum, n_select = self.get_params(col, options)

        starts = self.get_start_pos(non_nan_sum, n_select, options["boost"])
        n_iterations = len(starts)

        dtype = np.float16 if is_fuzzy else bool
        slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
        probs = np.zeros((n_iterations, nan_sum))
        weights = np.zeros((n_iterations, nan_sum))

        # START LOOP
        for i, start in enumerate(starts):
            end = min(start + n_select, non_nan_sum - 1)
            min_val, max_val = X[indices[start]], X[indices[end]]

            # update weights based on imputed values and normal distribution
            if is_fuzzy and not options["boost"]:
                if self.params["weight_approach"] == "imputed":
                    center = (min_val + max_val) / 2
                    X_dist = self.get_X_dist(col, center, nans)

                    w = options["n_select"] - n_select
                    r = min(0.25, max(0.05, (max_val - min_val) / 2))
                    weights[i, :] = self.get_weight(X_dist, w, r, nan_sum)
                probs[i, :] = self.get_probs(min_val, max_val)

            idx = indices[start:end]
            slices[i, idx] = True

        slices[:, nans] = self.update_nans(options, probs, weights)
        return slices

    def remove_nans(self, values, value_dict):
        if "?" in value_dict:
            index = np.where(values == "?")[0]
            return np.delete(values, index)
        return values

    def get_nom_dist(self, col, selected):
        nans = self.nans[col]
        X = self.X_complete[col].values[nans]
        X_dist = np.isin(X, selected).astype(float)
        n_dev, dev = 0.01, 0.1
        noise = np.random.normal(0, n_dev, len(X))
        X_dist += np.clip(noise, -dev, dev)
        return X_dist

    def get_categorical_slices(self, X, cache, col, **options):
        is_fuzzy = self.params["approach"] == "fuzzy"
        n_iterations = self.params["contrast_iterations"]
        nans, nan_sum, non_nan_sum, n_select = self.get_params(col, options)

        values, counts = np.unique(
            X, return_counts=True) if cache is None else cache["unique"]

        value_dict = dict(zip(values, counts))
        index_dict = {val: np.where(X == val)[0] for val in values}
        values = self.remove_nans(values, value_dict)

        probs = np.zeros((n_iterations, nan_sum))
        weights = np.zeros((n_iterations, nan_sum))
        dtype = np.float16 if is_fuzzy else bool

        if options["boost"]:
            s_per_class = 3 if len(values) < 10 else 1
            n_iterations = len(values) * s_per_class
            slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
            for i, value in enumerate(values):
                for j in range(s_per_class):
                    perm = np.random.permutation(value_dict[value])[:n_select]
                    idx = index_dict[value][perm]
                    slices[i * s_per_class + j, idx] = True

        else:
            slices = np.zeros((n_iterations, X.shape[0]), dtype=dtype)
            w = options["n_select"] - n_select
            for i in range(n_iterations):
                selected, cumsum = [], 0
                values = np.random.permutation(values)
                for v in values:
                    selected.append(v)
                    # update probs and weights
                    if is_fuzzy and self.params["weight_approach"] == "proba":
                        probs[i, :] += value_dict[v] / non_nan_sum

                    cumsum += value_dict[v]
                    if cumsum >= n_select:
                        n_missing = n_select - (cumsum - value_dict[v])
                        perm = np.random.permutation(value_dict[v])[:n_missing]
                        idx = index_dict[v][perm]
                        slices[i, idx] = True
                        break
                    slices[i, index_dict[v]] = True

                if is_fuzzy and self.params["weight_approach"] == "imputed":
                    X_dist = self.get_nom_dist(col, selected)
                    weights[i, :] = self.get_weight(X_dist, w, 0.1, nan_sum)

        slices[:, nans] = self.update_nans(options, probs, weights)
        return slices
