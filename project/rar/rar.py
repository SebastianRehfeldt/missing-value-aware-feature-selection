"""
    RaR class for feature selection
"""
import numpy as np
from math import factorial, ceil, log

from project.base import Subspacing
from .optimizer import deduce_relevances
from .rar_utils import sort_redundancies_by_target
from .rar_utils import get_ranking_arvind, get_ranking_tom


class RaR(Subspacing):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        self._update_params(**kwargs)
        self.hics = None

    def _update_params(self, **kwargs):
        # TODO: create RaR Params class and config
        alpha = kwargs.get("alpha", self._get_alpha())
        beta = kwargs.get("beta", 0.01)
        boost = kwargs.get("boost", 0.1)
        n_targets = kwargs.get("n_targets", 1)
        weight = kwargs.get("weight", 1)
        eval_method = kwargs.get("eval_method", "rar")
        approach = kwargs.get("approach", "deletion")
        create_category = kwargs.get("create_category", False)
        active_sampling = kwargs.get("active_sampling", True)
        min_slices = kwargs.get("min_slices", 10)
        min_samples = kwargs.get("min_samples", 3)
        resamples = kwargs.get("resamples", 5)
        max_subspaces = kwargs.get("max_subspaces", 1000)
        subspace_size = kwargs.get("subspace_size", self._get_size())
        subspace_method = kwargs.get("subspace_method", "adaptive")
        imputation_method = kwargs.get("imputation_method", "knn")
        contrast_iterations = kwargs.get("contrast_iterations", 100)
        redundancy_approach = kwargs.get("redundancy_approach", "arvind")

        self.params.update({
            "alpha": alpha,
            "beta": beta,
            "boost": boost,
            "n_targets": n_targets,
            "weight": weight,
            "eval_method": eval_method,
            "approach": approach,
            "create_category": create_category,
            "active_sampling": active_sampling,
            "min_slices": min_slices,
            "min_samples": min_samples,
            "resamples": resamples,
            "max_subspaces": max_subspaces,
            "subspace_size": subspace_size,
            "subspace_method": subspace_method,
            "imputation_method": imputation_method,
            "contrast_iterations": contrast_iterations,
            "redundancy_approach": redundancy_approach,
        })

        n_subspaces = self._get_n_subspaces()
        self.params["n_subspaces"] = kwargs.get("n_subspaces", n_subspaces)
        use_cache = self.should_enable_cache()
        self.params["cache_enabled"] = kwargs.get("cache_enabled", use_cache)

    def _get_alpha(self):
        # make sure to have enough samples inside a slice
        min_samples = 20
        if self.shape[0] == 0:
            return 0
        return max(0.01, min_samples / self.shape[0])

    def _get_size(self):
        # small change to rar to enable datasets with less than 5 features
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
        l = self.shape[1]
        s = min(3, k)

        def _choose(n, k):
            return factorial(n) // factorial(k) // factorial(n - k)

        denominator = log(1 - _choose(l - s, k - s) / _choose(l, k))
        return ceil(log(beta) / denominator)

    def _get_n_subspaces_linear(self):
        return self.shape[1] * self.params["contrast_iterations"]

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
        expected_size = all_slices * n_iterations * self.shape[0]
        if self.params["approach"] == "fuzzy":
            expected_size *= 2
        return False if expected_size > 1e9 else True

    def _evaluate_subspace(self, subspace):
        """
        Evaluate a subspace using hics measure

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        targets = []
        if self.params["redundancy_approach"] == "tom":
            open_features = [n for n in self.names if n not in subspace]
            n_targets = min(len(open_features), self.params["n_targets"])
            targets = np.random.choice(open_features, n_targets, False)

        results = self.hics.evaluate_subspace(subspace, targets)
        rel, red_s, is_empty, deviation = results

        if is_empty:
            rel, red_s, targets, deviation = 0, [], [], 0

        return {
            "relevance": rel,
            "deviation": deviation,
            "redundancies": red_s,
            "targets": targets,
        }

    def _deduce_feature_importances(self, knowledgebase):
        """
        Deduce single feature importances based on subspace results

        Arguments:
            knowledgebase {list} -- List of subspace results
        """
        relevances = deduce_relevances(self.names, knowledgebase)
        n_targets = self.params["n_targets"]

        if self.params["boost"] > 0:
            for key, value in relevances.items():
                alpha = self.params["boost"]
                boost = self.hics.get_boost(key)
                relevances[key] = (1 - alpha) * value + alpha * boost[0]

        # return ranking based on relevances only
        if n_targets == 0:
            return sorted(
                relevances.items(),
                key=lambda k_v: k_v[1],
                reverse=True,
            )

        # combine relevances with redundancies as done by tom or arvind
        if self.params["redundancy_approach"] == "tom":
            redundancies = sort_redundancies_by_target(knowledgebase)
            return get_ranking_tom(relevances, redundancies, self.names)

        return get_ranking_arvind(self.hics, relevances, self.names, n_targets)
