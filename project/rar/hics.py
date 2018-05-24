import numpy as np
from .contrast import calculate_contrast
from .slicing import get_slices


class HICS():
    def __init__(self, data, **params):
        self.data = data
        self.params = params

    def evaluate_subspace(self, names, types, target):
        # TODO: deletion with target removes more samples than neccessary
        # TODO: increase iterations when removing or imputing data
        # TODO: implement imputation
        new_X, new_y, new_t = self._complete(names, types, target)

        relevances = []
        redundancies = []
        # TODO make param for #iterations
        # TODO values from paper
        # TODO reduce slices by similarity
        n_select = int(0.8 * new_X.shape[0])
        for i in range(100):
            slice_ = get_slices(new_X, types, n_select)
            relevances.append(
                calculate_contrast(new_y, self.data.l_type, slice_))
            redundancies.append(
                calculate_contrast(new_t, self.data.f_types[target], slice_))

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
