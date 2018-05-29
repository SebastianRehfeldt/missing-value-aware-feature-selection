import numpy as np
from .contrast import calculate_contrasts
from .slicing import get_slices

from time import time


class HICS():
    def __init__(self, data, **params):
        self.data = data
        self.params = params
        # TODO: HICS should also work without target

    def evaluate_subspace(self, names, types, target):
        X, y, t = self._complete(names, types, target)
        l_type, t_type = self.data.l_type, self.data.f_types[target]

        # TODO #iterations should be a param
        # TODO increase iterations when having many missing values
        # TODO n_select should depend on #dimensions
        # TODO increase relevance if missingness is predictive
        n_select = int(0.8 * X.shape[0])

        slices = get_slices(X, types, n_select=n_select, n_iterations=100)
        c_cache = self._create_cache(y, l_type)
        t_cache = self._create_cache(t, t_type)

        # TODO: only return relevance if no target is specified
        # TODO: check if kld or ks are > 1 (but no normalization for now)
        # TODO: different value ranges from tests?
        # use 1-exp(-KLD(P,Q)) to normalize kld
        start = time()
        relevances = calculate_contrasts(y, l_type, slices, c_cache)
        print("Relevance (KLD)", time() - start)
        start = time()
        redundancies = calculate_contrasts(t, t_type, slices, t_cache)
        print("Redundancy (KS)", time() - start)
        return np.mean(relevances), np.mean(redundancies)

    def _create_cache(self, y, y_type):
        if y_type == "nominal":
            values, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return {
                "values": values,
                "probs": probs,
            }
        else:
            return {"sorted": np.sort(y)}

    def _complete(self, names, types, target):
        # TODO: implement imputation
        # TODO: HICS should also work without target
        # TODO: deletion should go into init_params from RaR
        if self.params.get("approach", "deletion") == "deletion":
            # TODO: 2-step deletion
            idx = self.data.X[names + [target]].notnull().apply(all, axis=1)
            new_X = self.data.X[names][idx]
            new_t = self.data.X[target][idx]
            new_y = self.data.y[idx]

        return new_X, new_y, new_t
