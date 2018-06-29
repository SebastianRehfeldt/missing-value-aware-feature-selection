import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices, combine_slices, prune_slices
from .slicing import get_categorical_slices, get_slices_num


class HICS():
    def __init__(self, data, nans, **params):
        self.data = data
        self.nans = nans
        self.params = params
        self._init_alphas()
        self._init_n_selects()

        if self.params["approach"] == "partial":
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
        max_subspace_size = self.params["subspace_size"][1]

        self.slices = {}
        for col in self.data.X:
            # TODO: sort the whole matrix at once
            # TODO: cache and reuse sorted indices
            # TODO: cache values and counts for categorical features
            # in partial approach
            # TODO: min samples should be a param and adapt to mr
            sorted_values = self.data.X[col].sort_values()

            self.slices[col] = {}
            for i in range(1, max_subspace_size + 1):
                options = {
                    "n_select": self.n_select_d[i],
                    "n_iterations": self.params["contrast_iterations"],
                    "approach": self.params["approach"],
                    "should_sample": self.params["sample_slices"],
                    "min_samples": 3,
                }
                if self.data.f_types[col] == "numeric":
                    self.slices[col][i] = get_slices_num(
                        self.data.X[col],
                        sorted_values.index,
                        self.nans[col],
                        **options,
                    )
                else:
                    self.slices[col][i] = get_categorical_slices(
                        self.data.X[col],
                        **options,
                    )

    def _cache_label(self):
        self.label_indices = np.argsort(self.data.y.values)
        self.label_values, self.label_counts = np.unique(
            self.data.y.values,
            return_counts=True,
        )

    def combine_slices(self, subspace):
        dim = len(subspace)
        slices = [self.slices[subspace[i]][dim] for i in range(dim)]
        slices = combine_slices(slices)

        max_nans = int(np.floor(dim / 2))
        nan_sums = self.nans[subspace].sum(1)
        slices[:, nan_sums > max_nans] = False
        options = {
            "n_iterations": self.params["contrast_iterations"],
            "min_samples": 3,
        }
        return prune_slices(slices, **options)

    def evaluate_subspace(self, subspace, targets=[]):
        # TODO: UNIFY
        if self.params["approach"] == "partial":
            slices, lengths = self.combine_slices(subspace)
            cache = self._create_cache(
                self.data.y,
                self.data.l_type,
                slices,
                lengths,
                self.label_indices,
            )
            relevances = calculate_contrasts(cache)
            relevance = np.mean(relevances)

            if np.isnan(relevance):
                return 0, [], True

            redundancies = self.compute_partial_redundancies(
                slices, lengths, targets)
            return 1 - np.exp(-1 * relevance), redundancies, False

        # Preparation
        types = self.data.f_types[subspace]
        X, y, indices, T = self._prepare_data(subspace, targets, types)
        if X.shape[0] < 10:
            return 0, [], True

        # Get slices
        slices = self.get_slices(X, types)
        if len(slices[0]) <= 5:
            return 0, [], True

        # Compute relevance and redundancy
        relevance = self.compute_relevance(slices, y)
        redundancies = self.compute_redundancies(slices, targets, indices, T)
        return relevance, redundancies, False

    def _prepare_data(self, subspace, targets, types):
        # TODO: check if nans exist (imputation would probably break)
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

    def compute_relevance(self, slices, y):
        cache = self._create_cache(y, self.data.l_type, *slices)
        relevances = calculate_contrasts(cache)
        return 1 - np.exp(-1 * np.mean(relevances))

    def compute_redundancies(self, slices, targets, indices, T):
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]
            if self.params["approach"] == "deletion":
                t_nans = self.nans[target][indices]
                t = self.data.X[target][indices][~t_nans]
                t_slices = slices[0][:, ~t_nans]

            if self.params["approach"] == "imputation":
                t = T[target]
                t_slices = slices[0]

            # TODO: update slice lengths!
            cache = self._create_cache(t, t_type, t_slices, slices[1])
            red_s = calculate_contrasts(cache)
            redundancies.append(np.mean(red_s))
        return redundancies

    def compute_partial_redundancies(self, slices, lengths, targets):
        # TODO: UNIFY WITH FUNCTION BEFORE
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]
            t_nans = self.nans[target]
            t = self.data.X[target][~t_nans]
            t_slices = slices[:, ~t_nans]
            # TODO: update slice lengths!
            cache = self._create_cache(t, t_type, t_slices, lengths)
            red_s = calculate_contrasts(cache)
            redundancies.append(np.mean(red_s))
        return redundancies

    def get_slices(self, X, types):
        # TODO: use this function from init slices when slicing is unified
        options = {
            "n_select": int(self.alphas_d[X.shape[1]] * X.shape[0]),
            "n_iterations": self.params["contrast_iterations"],
            "approach": self.params["approach"],
            "should_sample": self.params["sample_slices"],
            "min_samples": 3,
        }
        return get_slices(X, types, **options)

    def _create_cache(self, y, y_type, slices, lengths, label_indices=None):
        if label_indices is None:
            sorted_indices = np.argsort(y.values)
        else:
            sorted_indices = label_indices

        sorted_y = y.values[sorted_indices]

        cache = {
            "type": y_type,
            "lengths": lengths,
            "sorted": sorted_y,
            "slices": slices[:, sorted_indices],
        }

        if y_type == "nominal":
            if self.params["approach"] == "partial" and self.data.y.name == y.name:
                values, counts = self.label_values, self.label_counts
            else:
                values, counts = np.unique(sorted_y, return_counts=True)

            cache.update({
                "values": values,
                "probs": counts / len(sorted_y),
            })
        return cache
