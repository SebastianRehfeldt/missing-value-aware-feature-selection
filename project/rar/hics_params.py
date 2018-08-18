import numpy as np
import pandas as pd


class HICSParams():
    def __init__(self, data, nans, missing_rates, **params):
        self.data = data
        self.nans = nans
        self.missing_rates = missing_rates
        self.params = params
        self._initialize()

    def _initialize(self):
        self._init_alphas()
        self._init_n_selects()
        self._init_X_complete()
        self._init_slice_options()
        self._init_cache()

    def _init_alphas(self):
        alpha = self.params["alpha"]
        self.alphas_d = {
            i: alpha**(1 / i)
            for i in range(1, self.params["subspace_size"][1] + 1)
        }

    def _init_n_selects(self):
        self.n_select_d = {
            i: int(self.alphas_d[i] * self.data.shape[0])
            for i in range(1, self.params["subspace_size"][1] + 1)
        }

    def _init_X_complete(self):
        used_for_weights = self.params["weight_approach"] == "new"
        uses_imputation = self.params["approach"] == "imputation"
        used_for_redundancy = self.params["redundancy_approach"] == "arvind"

        if used_for_weights or (used_for_redundancy and uses_imputation):
            from project.utils.imputer import Imputer

            strategy = self.params["imputation_method"]
            imputer = Imputer(self.data.f_types, strategy)
            self.X_complete = imputer._complete(self.data.X)

    def _init_slice_options(self):
        self.slice_options = {
            "n_iterations": self.params["contrast_iterations"],
            "min_samples": self.params["min_samples"],
            "approach": self.params["approach"],
            "weight": self.params["weight"],
            "weight_approach": self.params["weight_approach"],
        }

    def _init_cache(self):
        self.label_indices = np.argsort(self.data.y.values)
        self.label_cache = np.unique(self.data.y.values, return_counts=True)

        uses_local_imputation = self.params["approach"] == "imputation"
        if self.params["cache_enabled"] and not uses_local_imputation:
            self._init_slice_cache()
            self._fill_slice_cache()

    def _init_slice_cache(self):
        dtype = np.float16 if self.params["approach"] == "fuzzy" else bool
        size = (self.data.shape[0], self.params["contrast_iterations"])
        self.slices = {
            col: {
                i: np.zeros(size, dtype=dtype).copy()
                for i in range(1, self.params["subspace_size"][1] + 1)
            }
            for col in self.data.X
        }

    def _fill_slice_cache(self):
        for col in self.data.X:
            X = self.data.X[col].to_frame()
            types = pd.Series(self.data.f_types[col], [col])
            f_cache = self.get_feature_cache(X, types, col)

            for i in range(1, self.params["subspace_size"][1] + 1):
                n = self.n_select_d[i]
                self.slices[col][i] = self.get_slices(X, types, f_cache, n, i)

    def get_feature_cache(self, X, types, col):
        if types[col] == "nominal":
            values, counts = np.unique(X, return_counts=True)
            return {
                "values": values,
                "counts": counts,
            }
        else:
            return {
                "nans": self.nans[col],
                "indices": np.argsort(self.data.X[col].values),
            }
