"""
    RKNN class for feature selection
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

from project.randomKNN.knn import KNN as knn_classifier
from project.shared.selector import Selector
from project.shared.subspacing import Subspacing
from project.utils.assertions import assert_series


class RKNN(Selector, Subspacing):

    def __init__(self, data, **kwargs):
        """
        RKNN Class

        Arguments:
            data {data} -- Data which should be fitted
        """
        super().__init__(data)

    def _init_parameters(self, parameters):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        self.params = {
            "n": parameters.get("n", int(self.data.shape[1]**2 / 2)),
            "m": parameters.get("m", int(np.sqrt(self.data.shape[1]))),
            "n_neighbors": parameters.get("n_neighbors", 3),
            "k": parameters.get("k", int(self.data.shape[1] / 2 + 1)),
            "nominal_distance": parameters.get("nominal_distance", 1),
            "use_cv": parameters.get("use_cv", False),
        }
        # TODO remove
        #self.params["n"] = min(self.params["n"], 10)
        #self.params["k"] = 7

    def calculate_feature_importances(self):
        """
        Calculate feature importances in different subspaces and combine them 
        """
        subspaces = self._get_unique_subscapes()
        score_map = self._evaluate_subspaces(subspaces)
        return self._deduce_feature_importances(score_map)

    def _evaluate_subspace(self, X, types):
        """
        Evaluate a subspace using knn

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        clf = knn_classifier(types, self.data.l_type,
                             n_neighbors=self.params["n_neighbors"])

        scoring = "accuracy" if self.data.l_type == "nominal" else "neg_mean_squared_error"
        if self.params["use_cv"]:
            cv = StratifiedKFold(self.data.y, n_folds=3, shuffle=True)
            scores = cross_val_score(
                clf, X, self.data.y, cv=cv, scoring=scoring)
            return np.mean(scores)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.data.y, test_size=0.5, stratify=self.data.y)
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
