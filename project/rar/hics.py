import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices


class HICS():
    def __init__(self, data, nans, **params):
        self.data = data
        self.nans = nans
        self.params = params
        self._init_alphas()

    def _init_alphas(self):
        alpha = self.params["alpha"]
        self.alphas_d = {
            i: alpha**(1 / i)
            for i in range(1, self.params["subspace_size"][1] + 1)
        }

    def evaluate_subspace(self, subspace, targets=[]):
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
        if self.params["approach"] == "deletion":
            X, y, indices, T = self._apply_deletion(subspace)

        if self.params["approach"] == "imputation":
            X, y, indices, T = self._apply_imputation(subspace, targets, types)
        return X, y, indices, T

    def _apply_deletion(self, subspace):
        nan_indices = np.sum(self.nans[subspace], axis=1) == 0
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

    def get_slices(self, X, types):
        options = {
            "slicing_method": self.params["slicing_method"],
            "n_iterations": self.params["contrast_iterations"],
            "n_select": int(self.alphas_d[X.shape[1]] * X.shape[0]),
        }
        return get_slices(X, types, **options)

    def _create_cache(self, y, y_type, slices, lengths):
        sorted_indices = np.argsort(y.values)
        sorted_y = y.values[sorted_indices]

        cache = {
            "type": y_type,
            "lengths": lengths,
            "sorted": sorted_y,
            "slices": slices[:, sorted_indices],
        }

        if y_type == "nominal":
            values, counts = np.unique(sorted_y, return_counts=True)
            cache.update({
                "values": values,
                "probs": counts / len(sorted_y),
            })
        return cache

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

            cache = self._create_cache(t, t_type, t_slices, slices[1])
            red_s = calculate_contrasts(cache)
            redundancies.append(1 - np.mean(red_s))
        return redundancies
