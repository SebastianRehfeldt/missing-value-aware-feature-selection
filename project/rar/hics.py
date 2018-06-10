import numpy as np
from .contrast import calculate_contrasts
from .slicing import get_slices


class HICS():
    def __init__(self, data, nans, **params):
        # TODO increase iterations when having many missing values?
        # TODO increase relevance if missingness is predictive
        # TODO: calculate alpha_d before and account for nans
        # TODO: implement imputation
        # TODO: cache sorted labels correctly (use argsort)
        # TODO: ignore subspaces with 0 slices
        self.data = data
        self.nans = nans
        self.params = params

    def evaluate_subspace(self, subspace, targets=[]):
        # Preparation
        types = self.data.f_types[subspace]
        if self.params["approach"] == "deletion":
            X, y, indices = self._apply_deletion(subspace)

        # Get slices
        n_iterations = self.params["contrast_iterations"]
        alpha_d = self.params["alpha"]**(1 / X.shape[1])
        n_select = int(alpha_d * X.shape[0])
        slices, lengths = get_slices(X, types, n_select, n_iterations)
        if len(slices) == 0:
            return 0, [0] * len(targets)

        # Compute relevance
        l_type = self.data.l_type
        l_cache = self._create_cache(y, l_type)
        relevances = calculate_contrasts(l_type, slices, l_cache, lengths)
        relevance = 1 - np.exp(-1 * np.mean(relevances))

        # Compute redundancies
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]
            if self.params["approach"] == "deletion":
                t_nans = self.nans[target][indices]
                t = self.data.X[target][indices][~t_nans]
                t_slices = slices[:, ~t_nans]

            t_cache = self._create_cache(t, t_type)
            red_s = calculate_contrasts(t_type, t_slices, t_cache, lengths)
            redundancies.append(1 - np.mean(red_s))

        return relevance, redundancies

    def _create_cache(self, y, y_type):
        sorted_y = np.sort(y)

        if y_type == "numeric":
            return {"sorted": sorted_y}
        else:
            values, counts = np.unique(sorted_y, return_counts=True)
            probs = counts / len(sorted_y)
            return {
                "values": values,
                "probs": probs,
                "sorted": sorted_y,
            }

    def _apply_deletion(self, subspace):
        nan_indices = np.sum(self.nans[subspace], axis=1) == 0
        new_X = self.data.X[subspace][nan_indices]
        new_y = self.data.y[nan_indices]
        return new_X, new_y, nan_indices
