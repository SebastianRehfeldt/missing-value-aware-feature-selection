import numpy as np
import pandas as pd
from .contrast import calculate_contrasts, calculate_contrasts2
from .slicing import get_slices, combine_slices, combine_slices2, prune_slices


class HICS():
    def __init__(self, data, nans, **params):
        self.data = data
        self.nans = nans
        self.params = params
        self._init_alphas()
        self._init_n_selects()

        if self.params["approach"] in ["partial", "fuzzy"]:
            self._init_slices()
            self._cache_label()

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

    def _init_slices(self):
        # TODO: fill cache on the fly
        self.slices = {}

        for col in self.data.X:
            X = self.data.X[col].to_frame()
            types = pd.Series(self.data.f_types[col], [col])

            sorted_indices = np.argsort(self.data.X[col].values)
            if types[col] == "nominal":
                values, counts = np.unique(X, return_counts=True)

            self.slices[col] = {}
            for i in range(1, self.params["subspace_size"][1] + 1):
                opts = {
                    "n_select": self.n_select_d[i],
                    "min_samples": 0,
                    "indices": sorted_indices,
                    "nans": self.nans[col],
                }
                if types[col] == "nominal":
                    opts.update({
                        "values": values,
                        "counts": counts,
                    })

                self.slices[col][i] = self.get_slices(X, types, **opts)[0]

    def _cache_label(self):
        self.label_indices = np.argsort(self.data.y.values)
        self.label_values, self.label_counts = np.unique(
            self.data.y.values,
            return_counts=True,
        )

    def evaluate_subspace(self, subspace, targets=[]):
        # GET SLICES
        if self.params["approach"] == "partial":
            y, indices, T = self.data.y, None, None
            slices, lengths = self.get_fault_tolerant_slices(subspace)
        elif self.params["approach"] == "fuzzy":
            y, indices, T = self.data.y, None, None
            slices, lengths = self.get_fuzzy_slices(subspace)
        else:
            types = self.data.f_types[subspace]
            X, y, indices, T = self._prepare_data(subspace, targets, types)
            if X.shape[0] < self.params["min_patterns"]:
                return 0, [], True
            slices, lengths = self.get_slices(X, types)

        # RETURN IF TOO FEW SLICES
        if len(slices) <= self.params["min_slices"]:
            return 0, [], True

        # COMPUTE RELEVANCE AND REDUNDANCIES
        rels = self.get_relevance(y, slices, lengths)
        reds = self.get_redundancies(slices, lengths, targets, indices, T)
        return rels, reds, False

    def get_relevance(self, y, slices, lengths):
        indices = None
        if self.params["approach"] in ["partial", "fuzzy"]:
            indices = self.label_indices

        l_type = self.data.l_type
        cache = self._create_cache(y, l_type, slices, lengths, indices)
        if self.params["approach"] == "fuzzy":
            relevances = calculate_contrasts2(cache)
        else:
            relevances = calculate_contrasts(cache)
        return 1 - np.exp(-1 * np.mean(relevances))

    def get_redundancies(self, slices, lengths, targets, indices=None, T=None):
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]
            if self.params["approach"] == "deletion":
                t_nans = self.nans[target][indices]
                t = self.data.X[target][indices][~t_nans]

            if self.params["approach"] in ["partial", "fuzzy"]:
                t_nans = self.nans[target]
                t = self.data.X[target][~t_nans]

            if self.params["approach"] in ["deletion", "partial", "fuzzy"]:
                min_samples = self.params["min_samples"]
                t_slices = slices[:, ~t_nans]
                t_slices, lengths = prune_slices(t_slices, min_samples)

            if self.params["approach"] == "imputation":
                t = T[target]
                t_slices = slices

            cache = self._create_cache(t, t_type, t_slices, lengths)
            if self.params["approach"] == "fuzzy":
                red_s = calculate_contrasts2(cache)
            else:
                red_s = calculate_contrasts(cache)
            redundancies.append(np.mean(red_s))
        return redundancies

    def _prepare_data(self, subspace, targets, types):
        if self.params["approach"] == "deletion":
            X, y, indices, T = self._apply_deletion(subspace)

        if self.params["approach"] == "imputation":
            X, y, indices, T = self._apply_imputation(subspace, targets, types)
        return X, y, indices, T

    def _apply_deletion(self, subspace):
        nan_indices = np.sum(self.nans[subspace].values, axis=1) == 0
        new_X = self.data.X[subspace][nan_indices]
        new_y = self.data.y[nan_indices]
        return new_X, new_y, nan_indices, None

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
        return X, self.data.y, None, T

    def get_slices(self, X, types, **opts):
        options = {
            "n_select": int(self.alphas_d[X.shape[1]] * X.shape[0]),
            "n_iterations": self.params["contrast_iterations"],
            "approach": self.params["approach"],
            "should_sample": self.params["sample_slices"],
            "min_samples": 3,
        }
        options.update(opts)
        return get_slices(X, types, **options)

    def get_fault_tolerant_slices(self, subspace):
        dim = len(subspace)
        slices = [self.slices[subspace[i]][dim] for i in range(dim)]
        slices = combine_slices(slices).copy()

        max_nans = int(np.floor(dim / 2))
        nan_sums = self.nans[subspace].sum(1)
        slices[:, nan_sums > max_nans] = False
        return prune_slices(slices, self.params["min_samples"])

    def get_fuzzy_slices(self, subspace):
        dim = len(subspace)
        slices = [self.slices[subspace[i]][dim] for i in range(dim)]
        slices = combine_slices2(slices).copy()
        return prune_slices(slices, self.params["min_samples"])

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
            y_name = self.data.y.name
            if self.params["approach"] in ["partial", "fuzzy"
                                           ] and y_name == y.name:
                values, counts = self.label_values, self.label_counts
            else:
                values, counts = np.unique(sorted_y, return_counts=True)

            cache.update({
                "values": values,
                "probs": counts / len(sorted_y),
            })
        return cache
