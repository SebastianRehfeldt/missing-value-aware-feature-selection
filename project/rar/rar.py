"""
    RaR class for feature selection
"""
import random
from time import time

from project.base import Subspacing
from .optimizer import deduce_relevances
from .hics import HICS
from .rar_utils import sort_redundancies_by_target, calculate_ranking


class RaR(Subspacing):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)

        # TODO: init params according to paper
        # TODO: set names in selector
        # TODO: make sure to get samples for each feature/target
        size = (1, min(5, int(self.shape[1] / 2)))
        self.params["subspace_size"] = kwargs.get("subspace_size", size)
        self.names = self.f_types.index.tolist()
        self.hics = None

    def _evaluate_subspace(self, X, types):
        """
        Evaluate a subspace using hics measure

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        # TODO: only pass names?
        names = X.columns.tolist()
        open_features = [name for name in self.names if name not in names]
        target = random.choice(open_features)

        # TODO: set hics earlier (override _evaluate_subspaces)
        if self.hics is None:
            self.hics = HICS(self.data, **self.params)

        start = time()
        rel, red = self.hics.evaluate_subspace(names, types, target)
        print("Evaluation: ", time() - start)
        return {
            "relevance": rel,
            "redundancy": red,
            "target": target,
        }

    def _deduce_feature_importances(self, knowledgebase):
        """
        Deduce single feature importances based on subspace results

        Arguments:
            knowledgebase {list} -- List of subspace results
        """
        relevances = deduce_relevances(self.names, knowledgebase)
        redundancies = sort_redundancies_by_target(knowledgebase)
        # TODO: cythonize ranking?
        return calculate_ranking(relevances, redundancies, self.names)
