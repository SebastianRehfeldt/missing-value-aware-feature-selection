"""
    SFS class for feature selection
"""
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

from project.randomKNN.knn import KNN as knn_classifier
from project.shared.selector import Selector
from project.tree.tree import Tree
from project.mutual_info.mutual_information import get_mutual_information


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
            "k": parameters.get("k", int(self.shape[1] / 2 + 1)),
            "nominal_distance": parameters.get("nominal_distance", 1),
            "use_cv": parameters.get("use_cv", False),
            "method": parameters.get("method", "knn"),
        }

    def calculate_feature_importances(self):
        """
        Calculate feature importances using sfs
        """
        score_map = {}
        open_features = self.data.X.columns.tolist()

        if self.params["method"] == "mi":
            self.data.add_salt()

        features = []
        while len(features) < self.params["k"]:
            scores = []
            for feature in open_features:
                if self.params["method"] == "mi":
                    s = self._evaluate_subspace_mi(features + [feature])
                else:
                    s = self._evaluate_subspace_clf(features + [feature])
                scores.append(s)

                if len(features) == 0:
                    score_map[feature] = s

            best_index = np.argsort(scores)[-1]
            next_f = open_features[best_index]
            features.append(next_f)
            open_features.remove(next_f)

        for f in open_features:
            score_map[f] = -1 * score_map[f]
        return score_map

    def _evaluate_subspace_mi(self, features):
        """
        Evaluate a subspace using knn

        Arguments:
        """
        X_sel, types = self.data.get_subspace(features)
        return get_mutual_information(X_sel, self.data.y, types,
                                      self.data.l_type,
                                      self.params["n_neighbors"])

    def _evaluate_subspace_clf(self, features):
        """
        Evaluate a subspace using knn

        Arguments:
        """
        X_sel, types = self.data.get_subspace(features)

        if self.params["method"] == "knn":
            clf = knn_classifier(
                types, self.l_type, n_neighbors=self.params["n_neighbors"])
        else:
            clf = Tree(self.data.to_table().domain)

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
