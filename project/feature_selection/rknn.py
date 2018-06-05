"""
    RKNN class for feature selection
"""
import numpy as np
from collections import defaultdict

from project.base import Subspacing
from project.shared.evaluation import evaluate_subspace


class RKNN(Subspacing):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        self.params["eval_method"] = kwargs.get("eval_method", "knn")

    def _evaluate_subspace(self, subspace):
        """
        Evaluate a subspace using knn

        Arguments:
            subspace {list(str)} -- List containing the selected feature names
        """
        X, types = self.data.get_subspace(subspace)
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
