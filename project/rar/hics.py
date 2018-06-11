import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices
from project.utils.imputer import Imputer


class HICS():
    def __init__(self, data, nans, **params):
        # TODO: increase iterations when having many missing values?
        # TODO: increase relevance if missingness is predictive
        # TODO: calculate alpha_d before and account for nans
        self.data = data
        self.nans = nans
        self.params = params

    def evaluate_subspace(self, subspace, targets=[]):
        # Preparation
        types = self.data.f_types[subspace]
        if self.params["approach"] == "deletion":
            X, y, indices = self._apply_deletion(subspace)

        if self.params["approach"] == "imputation":
            target_types = [self.data.f_types[t] for t in targets]
            target_types = pd.Series(target_types, index=targets)
            all_types = pd.concat([types, target_types], axis=0)
            cols = np.hstack((subspace, targets))

            imputer = Imputer(all_types, "knn")
            X_complete = imputer._complete(self.data.X[cols])
            T = X_complete[targets]
            X = X_complete[subspace]
            y = self.data.y

        if X.shape[0] < 10:
            return 0, [], True

        # Get slices
        n_iterations = self.params["contrast_iterations"]
        alpha_d = self.params["alpha"]**(1 / X.shape[1])
        n_select = int(alpha_d * X.shape[0])
        slices, lengths = get_slices(X, types, n_select, n_iterations)
        if len(slices) <= 5:
            return 0, [], True

        # Compute relevance
        cache = self._create_cache(y, self.data.l_type, slices, lengths)
        relevances = calculate_contrasts(cache)
        relevance = 1 - np.exp(-1 * np.mean(relevances))

        # Compute redundancies
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]
            if self.params["approach"] == "deletion":
                t_nans = self.nans[target][indices]
                t = self.data.X[target][indices][~t_nans]
                t_slices = slices[:, ~t_nans]

            if self.params["approach"] == "imputation":
                t = T[target]
                t_slices = slices

            cache = self._create_cache(t, t_type, t_slices, lengths)
            red_s = calculate_contrasts(cache)
            redundancies.append(1 - np.mean(red_s))

        return relevance, redundancies, False

    def _create_cache(self, y, y_type, slices, lengths):
        sorted_indices = np.argsort(y.values)
        sorted_y = y.values[sorted_indices]

        cache = {
            "type": y_type,
            "lengths": lengths,  # no need to sort
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

    def _apply_deletion(self, subspace):
        nan_indices = np.sum(self.nans[subspace], axis=1) == 0
        new_X = self.data.X[subspace][nan_indices]
        new_y = self.data.y[nan_indices]
        return new_X, new_y, nan_indices
