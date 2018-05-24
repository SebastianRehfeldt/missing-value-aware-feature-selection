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
        # TODO: increase iterations when removing or imputing data
        # TODO: implement imputation
        X, y, t = self._complete(names, types, target)
        l_type, t_type = self.data.l_type, self.data.f_types[target]

        # TODO make param for #iterations
        # TODO values from paper
        # TODO reduce slices by similarity
        relevances, redundancies = [], []
        n_select = int(0.8 * X.shape[0])
        for i in range(100):
            slice_ = get_slices(X, types, n_select)
            relevances.append(calculate_contrast(y, y[slice_], l_type))
            redundancies.append(calculate_contrast(t, t[slice_], t_type))

        # TODO: normalization?
        return np.mean(relevances), np.mean(redundancies)

    def _complete(self, names, types, target):
        if self.params.get("approach", "deletion") == "deletion":
            indices = self.data.X[names + [target]].notnull().apply(
                all, axis=1)
            new_X = self.data.X[names][indices]
            new_t = self.data.X[target][indices]
            new_y = self.data.y[indices]

        return new_X, new_y, new_t
