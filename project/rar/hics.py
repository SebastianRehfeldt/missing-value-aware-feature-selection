import numpy as np
import pandas as pd
from .contrast import calculate_contrasts
from .slicing import get_slices

from time import time


class HICS():
    def __init__(self, data, **params):
        # TODO: HICS should also work without target
        self.data = data
        self.params = params

    def evaluate_subspace(self, names, types, target):
        # TODO increase iterations when having many missing values
        # TODO increase relevance if missingness is predictive
        # TODO: check if kld or ks are > 1 (but no normalization for now)
        # TODO: cythonize contrast?
        # TODO: different value ranges from tests?
        # use 1-exp(-KLD(P,Q)) to normalize kld
        start = time()
        X, y, t = self._complete(names, types, target)
        print("Complete", time() - start)
        l_type, t_type = self.data.l_type, self.data.f_types[target]

        n_iterations = self.params["contrast_iterations"]
        alpha_d = self.params["alpha"]**(1 / X.shape[1])
        n_select = int(alpha_d * X.shape[0])

        start = time()
        slices = get_slices(X, types, n_select, n_iterations)
        print("Slicing", time() - start)

        start = time()
        c_cache = self._create_cache(y, l_type)
        t_cache = self._create_cache(t, t_type)
        print("Caching", time() - start)

        start = time()
        relevances = calculate_contrasts(l_type, slices, c_cache)
        print("Relevances (KLD)", time() - start)

        start = time()
        redundancies = calculate_contrasts(t_type, slices, t_cache)
        print("Redundancies (KS)", time() - start)
        return pd.Series(relevances).mean(), np.mean(redundancies)

    def _create_cache(self, y, y_type):
        sorted_y = np.sort(y)
        values, counts = np.unique(sorted_y, return_counts=True)
        probs = counts / len(sorted_y)
        return {
            "values": values,
            "probs": probs,
            "sorted": sorted_y,
        }

    def _complete(self, names, types, target):
        # TODO: implement imputation
        # TODO: 2-step deletion
        if self.params["approach"] == "deletion":
            idx = self.data.X[names + [target]].notnull().apply(all, axis=1)
            new_X = self.data.X[names][idx]
            new_t = self.data.X[target][idx]
            new_y = self.data.y[idx]

        return new_X, new_y, new_t
