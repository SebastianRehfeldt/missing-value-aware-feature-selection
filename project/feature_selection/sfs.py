"""
    SFS class for feature selection
"""
import numpy as np

from project.base import Selector
from project.shared import evaluate_subspace


class SFS(Selector):
    def __init__(self, f_types, l_type, shape, **kwargs):
        """
        RKNN Class

        Arguments:
            f_types {pd.Series} -- Series containing feature types
            l_type {str} -- Type of label
            shape {tuple} -- Shape of feature matrix
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
            "mi_neighbors": parameters.get("mi_neighbors", 6),
            "k": parameters.get("k", int(self.shape[1] / 2 + 1)),
            "nominal_distance": parameters.get("nominal_distance", 1),
            "use_cv": parameters.get("use_cv", False),
            "method": parameters.get("method", "mi"),
        }

    def _fit(self):
        """
        Calculate feature importances using sfs
        """
        if self.params["method"] == "mi":
            self.data.add_salt()

        score_map = {}
        open_features = self.data.X.columns.tolist()

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
