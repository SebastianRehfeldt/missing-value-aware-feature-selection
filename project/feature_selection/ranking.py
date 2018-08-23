import numpy as np
from project.base import Selector
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.information_theoretical_based.FCBF import fcbf
from skfeature.function.statistical_based.CFS import cfs
from project.feature_selection.reliefF import reliefF as myrelief


class Ranking(Selector):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        self.params["eval_method"] = kwargs.get("eval_method", "relief")

    def _fit(self):
        method = self.params["eval_method"]
        X, y, d = self.data.X.values, self.data.y.values, self.shape[1]
        X = np.nan_to_num(X)
        X = np.round(X, 2)

        if method == "mrmr":
            sorted_indices = mrmr(X, y, mode="index", n_selected_features=d)
        elif method == "myrelief":
            dist_params = {
                "nominal_distance": self.params["nominal_distance"],
                "f_types": self.data.f_types.values,
            }
            sorted_indices = myrelief(X, y, dist_params, mode="index")
        else:
            sorted_indices = {
                "relief": reliefF,
                "fcbf": fcbf,
                "cfs": cfs,
            }[method](
                X, y, mode="index")

        self.feature_importances = dict.fromkeys(self.feature_importances, 0)

        for i, idx in enumerate(sorted_indices):
            feature = self.names[idx]
            self.feature_importances[feature] = 1 / (i + 1)
