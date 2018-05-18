"""
    RKNN class for feature selection
"""
import numpy as np
from collections import defaultdict

from project.base import Subspacing
from project.shared import evaluate_subspace


class RKNN(Subspacing):
    def __init__(self, f_types, l_type, shape, **kwargs):
        """
        RKNN Class

        Arguments:
            f_types {pd.Series} -- Series containing feature types
            l_type {str} -- Type of label
        """
        super().__init__(f_types, l_type, shape, **kwargs)

    def _init_parameters(self, parameters):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        self.params = {
            "n": parameters.get("n", int(self.shape[1]**2 / 2)),
            "m": parameters.get("m", int(np.sqrt(self.shape[1]))),
            "n_neighbors": parameters.get("n_neighbors", 3),
            "k": parameters.get("k", int(self.shape[1] / 2 + 1)),
            "nominal_distance": parameters.get("nominal_distance", 1),
            "use_cv": parameters.get("use_cv", False),
            "method": parameters.get("method", "knn"),
        }
        # TODO remove
        # self.params["n"] = min(self.params["n"], 10)
        # self.params["k"] = 7

    def _evaluate_subspace(self, X, types):
        """
        Evaluate a subspace using knn

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        return evaluate_subspace(X, self.data.y, types, self.l_type,
                                 self.domain, **self.params)

    def _deduce_feature_importances(self, knowledgebase):
        """
        Deduce single feature importances based on subspace results

        Arguments:
            knowledgebase {list} -- List of subspace results
        """
        score_map = defaultdict(list)
        for subspace in knowledgebase:
            for feature in subspace["features"]:
                score_map[feature].append(subspace["score"])

        return dict((k, np.mean(v)) for k, v in score_map.items())
