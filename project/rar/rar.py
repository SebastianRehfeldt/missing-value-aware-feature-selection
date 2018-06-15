"""
    RaR class for feature selection
"""
import numpy as np
from math import factorial, ceil, log

from project.base import Subspacing
from .optimizer import deduce_relevances
from .rar_utils import sort_redundancies_by_target, calculate_ranking


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
        alpha = kwargs.get("alpha", self._get_alpha())
        beta = kwargs.get("beta", 0.01)
        n_targets = kwargs.get("n_targets", 3)
        eval_method = kwargs.get("eval_method", "rar")
        approach = kwargs.get("approach", "deletion")
        use_pearson = kwargs.get("use_pearson", True)
        max_subspaces = kwargs.get("max_subspaces", 1000)
        subspace_size = kwargs.get("subspace_size", self._get_size())
        slicing_method = kwargs.get("slicing_method", "mating")
        subspace_method = kwargs.get("subspace_method", "adaptive")
        imputation_method = kwargs.get("imputation_method", "knn")
        contrast_iterations = kwargs.get("contrast_iterations", 100)

        self.params.update({
            "alpha": alpha,
            "beta": beta,
            "n_targets": n_targets,
            "eval_method": eval_method,
            "approach": approach,
            "use_pearson": use_pearson,
            "max_subspaces": max_subspaces,
            "subspace_size": subspace_size,
            "slicing_method": slicing_method,
            "subspace_method": subspace_method,
            "imputation_method": imputation_method,
            "contrast_iterations": contrast_iterations,
        })

        n_subspaces = self._get_n_subspaces()
        self.params["n_subspaces"] = kwargs.get("n_subspaces", n_subspaces)

    def _get_alpha(self):
        # make sure to have enough samples inside a slice
        min_samples = 30
        return max(0.01, min_samples / self.shape[0])

    def _get_size(self):
        # small change to rar to enable datasets with less than 5 features
        max_size = int(self.shape[1] / 2)
        return (1, min(5, max_size))

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

    def _evaluate_subspace(self, subspace):
        """
        Evaluate a subspace using hics measure

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        open_features = [n for n in self.names if n not in subspace]
        n_targets = min(len(open_features), self.params["n_targets"])
        targets = np.random.choice(open_features, n_targets, False)

        rel, red_s, is_empty = self.hics.evaluate_subspace(subspace, targets)

        if is_empty:
            return {
                "relevance": 0,
                "redundancies": [],
                "targets": [],
            }
        else:
            return {
                "relevance": rel,
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
        self.relevances = relevances
        redundancies = sort_redundancies_by_target(knowledgebase)
        self.redundancies = redundancies

        redundancies_1d = None
        if self.params["use_pearson"]:
            redundancies_1d = self.data.X.corr().fillna(0)
        return calculate_ranking(relevances, redundancies, redundancies_1d,
                                 self.names)
