import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices, combine_slices, prune_slices


class HICS():
    def __init__(self, data, nans, missing_rates, **params):
        self.data = data
        self.nans = nans
        self.missing_rates = missing_rates
        self.params = params
        self._init_alphas()
        self._init_n_selects()
        self._init_slice_options()
        self._cache_label()

        if self.params["weight_approach"] == "new":
            from project.utils.imputer import Imputer

            imputer = Imputer(self.data.f_types, "mice")
            self.X_complete = imputer._complete(self.data.X)

        if not self.params["approach"] == "imputation" and self.params["cache_enabled"]:
            self._init_slice_cache()
            self._fill_slice_cache()

    def _init_alphas(self):
        alpha = self.params["alpha"]
        self.alphas_d = {
            i: alpha**(1 / i)
            for i in range(1, self.params["subspace_size"][1] + 1)
        }

    def _init_n_selects(self):
        self.n_select_d = {
            i: int(self.alphas_d[i] * self.data.shape[0])
            for i in range(1, self.params["subspace_size"][1] + 1)
        }

    def _init_slice_options(self):
        self.slice_options = {
            "n_iterations": self.params["contrast_iterations"],
            "min_samples": self.params["min_samples"],
            "approach": self.params["approach"],
            "weight": self.params["weight"],
            "weight_approach": self.params["weight_approach"],
        }

    def _cache_label(self):
        self.label_indices = np.argsort(self.data.y.values)
        self.label_values, self.label_counts = np.unique(
            self.data.y.values,
            return_counts=True,
        )

    def _init_slice_cache(self):
        dtype = np.float16 if self.params["approach"] == "fuzzy" else bool
        size = (self.data.shape[0], self.params["contrast_iterations"])
        self.slices = {
            col: {
                i: np.zeros(size, dtype=dtype).copy()
                for i in range(1, self.params["subspace_size"][1] + 1)
            }
            for col in self.data.X
        }

    def _fill_slice_cache(self):
        for col in self.data.X:
            X = self.data.X[col].to_frame()
            types = pd.Series(self.data.f_types[col], [col])

            if types[col] == "nominal":
                values, counts = np.unique(X, return_counts=True)
                cache = {
                    "values": values,
                    "counts": counts,
                }
            else:
                cache = {
                    "nans": self.nans[col],
                    "indices": np.argsort(self.data.X[col].values),
                }

            for i in range(1, self.params["subspace_size"][1] + 1):
                n = self.n_select_d[i]
                self.slices[col][i] = self.get_slices(X, types, cache, n, i)

    def get_slices(self, X, types, cache=None, n=None, d=None, b=False):
        options = self.slice_options.copy()
        options["n_select"] = n or self.n_select_d[X.shape[1]]
        options["alpha"] = self.alphas_d[X.shape[1]]
        options["d"] = d or X.shape[1]
        options["boost"] = b
        if self.params["weight_approach"] == "new":
            options["X_complete"] = self.X_complete[X.columns]
        return get_slices(X, types, cache, **options)

    def get_cached_slices(self, subspace, use_cache=True):
        dim = len(subspace)

        if self.params["cache_enabled"] and use_cache:
            slices = [self.slices[subspace[i]][dim] for i in range(dim)]
            slices = combine_slices(slices).copy()
        else:
            X = self.data.X[subspace]
            types = self.data.f_types[subspace]
            slices = self.get_slices(X, types)

        min_samples = self.params["min_samples"]
        if self.params["approach"] in ["partial", "fuzzy"]:
            max_nans = int(np.floor(dim / 2))
            if self.params["approach"] == "fuzzy":
                #min_samples = 0
                max_nans = dim

            nan_sums = np.sum(self.nans[subspace].values, axis=1)
            slices[:, nan_sums > max_nans] = False
        return prune_slices(slices, min_samples)

    def get_boost(self, feature):
        X = self.data.X[feature].to_frame()
        types = pd.Series(self.data.f_types[feature], [feature])
        slices = self.get_slices(X, types, n=10, b=True)
        lengths = np.sum(slices, axis=1)
        return self.get_relevance(slices, lengths)

    def evaluate_subspace(self, subspace, targets=[]):
        # GET SLICES
        T = None
        if self.params["approach"] == "imputation":
            types = self.data.f_types[subspace]
            X, T = self._apply_imputation(subspace, targets, types)
            slices = self.get_slices(X, types)
            slices, lengths = prune_slices(slices, self.params["min_samples"])
        else:
            slices, lengths = self.get_cached_slices(subspace)

        # RETURN IF TOO FEW SLICES
        if len(slices) <= self.params["min_slices"]:
            return 0, [], True, 0

        # COMPUTE RELEVANCE AND REDUNDANCIES
        rels, deviations = self.get_relevance(slices, lengths)
        n_resamples = self.params["resamples"]
        if deviations > 0.1 and len(subspace) == 1 and n_resamples > 0:
            n_retries = int(np.floor(deviations * n_resamples / 0.5)) + 1
            n_resamples = min(n_resamples, n_retries)
            resamples = np.zeros(n_resamples + 1)
            resamples[0] = rels
            for i in range(1, n_resamples + 1):
                new_slices, new_lengths = self.get_cached_slices(
                    subspace, False)
                resamples[i] = self.get_relevance(new_slices, new_lengths)[0]
            rels = np.mean(resamples)

        reds = self.get_redundancies(slices, lengths, targets, T)
        return rels, reds, False, deviations

    def get_relevance(self, slices, lengths):
        y = self.data.y
        l_type = self.data.l_type
        indices = self.label_indices
        cache = self._create_cache(y, l_type, slices, lengths, indices)
        relevances = calculate_contrasts(cache)
        rel = 1 - np.exp(-1 * np.mean(relevances))
        return (rel, np.std(relevances))

    def get_redundancies(self, slices, lengths, targets, T=None):
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]

            if self.params["approach"] == "imputation":
                # impute nans in redundancy target
                t = T[target]
                t_slices = slices
            else:
                # remove samples with nans in target for redundancy calculation
                t_nans = self.nans[target]
                if np.sum(t_nans.values) > 0:
                    t = self.data.X[target][~t_nans]

                    min_samples = self.params["min_samples"]
                    t_slices = slices[:, ~t_nans]
                    t_slices, lengths = prune_slices(t_slices, min_samples)
                else:
                    t = self.data.X[target]
                    t_slices = slices

            if len(t_slices) <= self.params["min_slices"]:
                redundancies.append(0)
            else:
                cache = self._create_cache(t, t_type, t_slices, lengths)
                red_s = calculate_contrasts(cache)
                redundancies.append(np.mean(red_s))
        return redundancies

    def _apply_imputation(self, subspace, targets, types):
        from project.utils.imputer import Imputer

        target_types = [self.data.f_types[t] for t in targets]
        target_types = pd.Series(target_types, index=targets)
        all_types = pd.concat([types, target_types], axis=0)
        cols = np.hstack((subspace, targets))

        imputer = Imputer(all_types, self.params["imputation_method"])
        X_complete = imputer._complete(self.data.X[cols])
        T = X_complete[targets]
        X = X_complete[subspace]
        return X, T

    def _create_cache(self, y, y_type, slices, lengths, cached_indices=None):
        if cached_indices is None:
            sorted_indices = np.argsort(y.values)
        else:
            sorted_indices = cached_indices

        sorted_y = y.values[sorted_indices]
        cache = {
            "type": y_type,
            "lengths": lengths,
            "sorted": sorted_y,
            "slices": slices[:, sorted_indices],
        }

        if y_type == "nominal":
            if self.data.y.name == y.name:
                values, counts = self.label_values, self.label_counts
            else:
                values, counts = np.unique(sorted_y, return_counts=True)

            cache.update({
                "values": values,
                "counts": counts,
            })
        return cache
