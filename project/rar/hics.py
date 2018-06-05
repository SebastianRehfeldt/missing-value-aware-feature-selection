import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices

from time import time


class HICS():
    def __init__(self, data, nans, **params):
        # TODO: HICS should also work without target
        self.data = data
        self.nans = nans
        self.params = params

    def evaluate_subspace(self, subspace, target):
        # TODO increase iterations when having many missing values?
        # TODO increase relevance if missingness is predictive
        # TODO: different value ranges from tests?
        # use 1-exp(-KLD(P,Q)) to normalize kld
        X, y, t = self._complete(subspace, target)
        l_type = self.data.l_type
        t_type = self.data.f_types[target]
        types = self.data.f_types[subspace]

        # TODO: calculate before and account for nans
        n_iterations = self.params["contrast_iterations"]
        alpha_d = self.params["alpha"]**(1 / X.shape[1])
        n_select = int(alpha_d * X.shape[0])

        start = time()
        slices = get_slices(X, types, n_select, n_iterations)
        if len(slices) == 0:
            return 0, 0
        #print("Slicing", time() - start)

        start = time()
        c_cache = self._create_cache(y, l_type)
        t_cache = self._create_cache(t, t_type)
        #print("Caching", time() - start)

        start = time()
        relevances = calculate_contrasts(l_type, slices, c_cache)
        #print("Relevances (KLD)", time() - start)

        start = time()
        redundancies = calculate_contrasts(t_type, slices, t_cache)
        #print("Redundancies (KS)", time() - start)
        #print(1 / 0)
        return np.mean(relevances), np.mean(redundancies)

    def _create_cache(self, y, y_type):
        sorted_y = np.sort(y)
        values, counts = np.unique(sorted_y, return_counts=True)
        probs = counts / len(sorted_y)
        return {
            "values": values,
            "probs": probs,
            "sorted": sorted_y,
        }

    def _complete(self, subspace, target):
        # TODO: implement imputation
        # TODO: 2-step deletion
        if self.params["approach"] == "deletion":
            idx = np.sum(self.nans[subspace + [target]], axis=1) == 0
            new_X = self.data.X[subspace][idx]
            new_t = self.data.X[target][idx]
            new_y = self.data.y[idx]

        return new_X, new_y, new_t
