"""
    SFS class for feature selection
"""
import numpy as np

from project.base import Selector
from project.shared.evaluation import evaluate_subspace


class SFS(Selector):
    def _init_parameters(self, **kwargs):
        super()._init_parameters(**kwargs)
        self.params["do_stop"] = kwargs.get("do_stop", False)

    def _fit(self):
        """
        Calculate feature importances using sfs
        """
        self.feature_importances = {}
        self.scores = []
        open_features = self.names[:]

        i, features = 0, []
        n = self.params["k"] if self.params["do_stop"] else self.data.shape[1]
        while len(features) < n:
            scores = []
            for feature in open_features:
                X_sel, types = self.data.get_subspace(features + [feature])
                score = evaluate_subspace(X_sel, self.data.y, types,
                                          self.l_type, self.domain,
                                          **self.params)
                scores.append(score)

            next_feature = open_features[np.argsort(scores)[-1]]
            features.append(next_feature)
            open_features.remove(next_feature)
            i += 1
            self.feature_importances[next_feature] = 1 / i
            self.scores.append(np.max(scores))
