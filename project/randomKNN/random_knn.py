"""
    RKNN class for feature selection
"""
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

from project.randomKNN.knn import KNN as knn_classifier
from project.shared.subspacing import Subspacing


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
        clf = knn_classifier(
            types, self.data.l_type, n_neighbors=self.params["n_neighbors"])

        scoring = "accuracy"
        stratify = self.data.y
        if self.data.l_type == "numeric":
            scoring = "neg_mean_squared_error"
            stratify = None

        if self.params["use_cv"]:
            cv = StratifiedKFold(self.data.y, n_folds=3, shuffle=True)
            scores = cross_val_score(
                clf, X, self.data.y, cv=cv, scoring=scoring)
            return np.mean(scores)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.data.y, test_size=0.5, stratify=stratify)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return {
                "accuracy": accuracy_score,
                "neg_mean_squared_error": mean_squared_error,
            }[scoring](y_test, y_pred)

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
