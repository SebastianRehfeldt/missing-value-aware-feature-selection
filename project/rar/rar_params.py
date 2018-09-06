import numpy as np
from math import factorial, ceil, log
from project.base import Subspacing


class RaRParams(Subspacing):
    def _init_parameters(self, **kwargs):
        super()._init_parameters(**kwargs)
        self._init_params(**kwargs)
        self.hics = None
        self.interactions = []
        self.relevances = {}

    def _init_boosts(self):
        self.params.update({
            "boost_value": 0.1,
            "boost_inter": 0.1,
            "boost_corr": 0.1,
            "active_sampling": True,
            "active_sampling_mr": True,
            "active_sampling_rel": True,
            "active_sampling_corr": True,
        })

    def _init_methods(self):
        self.params.update({
            "eval_method": "rar",
            "approach": "deletion",
            "create_category": False,
            "weight_approach": "alpha",
            "dist_method": "distance",
            "imputation_method": "knn",
            "subspace_method": "adaptive",
            "redundancy_approach": "arvind",
        })

    def _init_constants(self):
        self.params.update({
            "weight": 1,
            "beta": 0.01,
            "n_targets": 1,
            "n_resamples": 5,
            "min_slices": 30,
            "regularization": 1,
            "max_subspaces": 2000,
            "subspace_size": self._get_size(),
        })

    def _init_params(self, **kwargs):
        self._init_boosts()
        self._init_methods()
        self._init_constants()
        self.params.update(**kwargs)

        n_subspaces = kwargs.get("n_subspaces", self._get_n_subspaces())
        self.params["n_subspaces"] = n_subspaces

    def update_params(self):
        n_classes = len(np.unique(self.data.y.values))
        a = self.params.get("alpha", self._get_alpha(n_classes))

        min_samples = int(np.clip(np.round(n_classes * 0.75), 5, 20))
        min_samples = self.params.get("min_samples", min_samples)

        n_iterations = np.clip(int(1.5 / a), 75, 125)
        n_iterations = self.params.get("contrast_iterations", n_iterations)

        self.params.update({
            "alpha": a,
            "n_classes": n_classes,
            "min_samples": min_samples,
            "contrast_iterations": n_iterations,
        })

    def _get_alpha(self, n_classes):
        alpha = (10 * n_classes / self.data.shape[0])**1.5
        return np.clip(alpha, 0.005, 0.2)

    def _get_size(self):
        max_size = int(self.shape[1] / 2)
        return (1, min(3, max_size))

    def _get_n_subspaces(self):
        n_subspaces = {
            "adaptive": self._get_n_subspaces_adaptive,
            "linear": self._get_n_subspaces_linear,
            "fixed": self._get_n_subspaces_fixed,
        }[self.params["subspace_method"]]()
        return min(self.params["max_subspaces"], n_subspaces)

    def _get_n_subspaces_adaptive(self):
        # see thesis of tom at page 42
        beta = self.params["beta"]
        k = self.params["subspace_size"][1]
        d = self.shape[1]
        s = min(3, k)

        def _choose(n, k):
            return factorial(n) // factorial(k) // factorial(n - k)

        denominator = log(1 - _choose(d - s, k - s) / _choose(d, k))
        return ceil(log(beta) / denominator)

    def _get_n_subspaces_linear(self):
        return self.shape[1] * 10 * self.params["subspace_size"][1]

    def _get_n_subspaces_fixed(self):
        return 1000

    def should_enable_cache(self):
        # CALCULATE EXPECTED SIZES WITHOUT CACHE
        subspace_size = self.params["subspace_size"]
        mean_dim = np.mean(subspace_size)
        max_slices = mean_dim * self.params["n_subspaces"]

        # SLICES IN CACHE
        n_dimensions = subspace_size[1] - subspace_size[0] + 1
        all_slices = n_dimensions * self.shape[1]

        if all_slices > max_slices:
            return False

        # ESTIMATE NEEDED SIZE FOR CACHE (MAX = 1G)
        n_iterations = self.params["contrast_iterations"]
        expected_size = all_slices * n_iterations * self.data.shape[0]
        if self.params["approach"] == "fuzzy":
            expected_size *= 2
        return False if expected_size > 1e9 else True
