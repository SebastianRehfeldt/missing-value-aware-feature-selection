"""
    SFS class for feature selection
"""
import numpy as np

from project.base import Selector
from project.shared import evaluate_subspace


class SFS(Selector):
    def _fit(self):
        """
        Calculate feature importances using sfs
        """
        score_map = {}
        open_features = self.names[:]

        features = []
        while len(features) < self.params["k"]:
            scores = []
            for feature in open_features:
                X_sel, types = self.data.get_subspace(features + [feature])
                score = evaluate_subspace(X_sel, self.data.y, types,
                                          self.l_type, self.domain,
                                          **self.params)
                scores.append(score)

                if len(features) == 0:
                    score_map[feature] = score

            next_feature = open_features[np.argsort(scores)[-1]]
            features.append(next_feature)
            open_features.remove(next_feature)

        for f in open_features:
            score_map[f] = -1 * score_map[f]
        self.feature_importances = score_map
