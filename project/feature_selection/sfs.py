"""
    SFS class for feature selection
"""
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

from project.base import Selector
from project.classifier import KNN, Tree
from project.shared import get_mutual_information


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
            "method": parameters.get("method", "knn"),
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
                score = self._evaluate(features + [feature])
                scores.append(score)

                if len(features) == 0:
                    score_map[feature] = score

            next_feature = open_features[np.argsort(scores)[-1]]
            features.append(next_feature)
            open_features.remove(next_feature)

        for f in open_features:
            score_map[f] = -1 * score_map[f]
        self.feature_importances = score_map

    def _evaluate(self, features):
        X_sel, types = self.data.get_subspace(features)
        return {
            "mi": self._evaluate_subspace_mi,
            "knn": self._evaluate_subspace_clf,
            "tree": self._evaluate_subspace_clf,
        }[self.params["method"]](X_sel, types)

    def _evaluate_subspace_mi(self, X_sel, types):
        """
        Evaluate a subspace using knn

        Arguments:
        """
        return get_mutual_information(X_sel, self.data.y, types,
                                      self.data.l_type,
                                      self.params["mi_neighbors"])

    def _evaluate_subspace_clf(self, X_sel, types):
        """
        Evaluate a subspace using knn

        Arguments:
        """
        if self.params["method"] == "knn":
            clf = KNN(
                types, self.l_type, n_neighbors=self.params["n_neighbors"])
        else:
            clf = Tree(self.domain)

        scoring = "accuracy"
        stratify = self.data.y
        if self.data.l_type == "numeric":
            scoring = "neg_mean_squared_error"
            stratify = None

        if self.params["use_cv"]:
            cv = StratifiedKFold(self.data.y, n_folds=3, shuffle=True)
            scores = cross_val_score(
                clf, X_sel, self.data.y, cv=cv, scoring=scoring)
            return np.mean(scores)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sel, self.data.y, test_size=0.5, stratify=stratify)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                "accuracy": accuracy_score,
                "neg_mean_squared_error": mean_squared_error,
            }[scoring](y_test, y_pred)
