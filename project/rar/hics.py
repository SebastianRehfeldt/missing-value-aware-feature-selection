import numpy as np
from .contrast import calculate_contrast
from .slicing import get_slices


class HICS():
    def __init__(self, data, **params):
        self.data = data
        self.params = params
        # TODO: HICS should also work without target

    def evaluate_subspace(self, names, types, target):
        # TODO: deletion with target removes more samples than neccessary
        # TODO: allow nan in target?
        # TODO: 2-step deletion
        X, y, t = self._complete(names, types, target)
        l_type, t_type = self.data.l_type, self.data.f_types[target]

        # TODO #iterations should be a param
        # TODO increase iterations when having many missing values
        # TODO n_select should depend on #dimensions
        # TODO reduce slices by similarity
        # TODO pairing from single slices
        # TODO get single feature slices and combine them here
        # TODO improve speed by using cython/ caching/ presorting/ preslicing
        # TODO increase relevance if missingness is predictive
        relevances, redundancies = [], []
        n_select = int(0.8 * X.shape[0])

        slices = get_slices(X, types, n_select=n_select, n_iterations=100)
        for slice_ in slices:
            relevances.append(calculate_contrast(y, y[slice_], l_type))
            # TODO: only return relevance if no target is specified
            redundancies.append(calculate_contrast(t, t[slice_], t_type))

        # TODO: check if kld or ks are > 1 (but no normalization for now)
        return np.mean(relevances), np.mean(redundancies)

    def _complete(self, names, types, target):
        # TODO: implement imputation
        # TODO: HICS should also work without target
        # TODO: deletion should go into init_params from RaR
        if self.params.get("approach", "deletion") == "deletion":
            idx = self.data.X[names + [target]].notnull().apply(all, axis=1)
            new_X = self.data.X[names][idx]
            new_t = self.data.X[target][idx]
            new_y = self.data.y[idx]

        return new_X, new_y, new_t
