"""
    RKNN class for feature selection
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold

from project.randomKNN.knn import KNN as knn_classifier
from project.shared.selector import Selector
from project.shared.subspacing import Subspacing
from project.utils.assertions import assert_df


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
            "k": parameters.get("k", 3),
            "method": parameters.get("method", "imputation"),
            "nominal_distance": parameters.get("nominal_distance", 1),
        }

    def calculate_feature_importances(self):
        """
        Calculate feature importances in different subspaces and combine them 
        """
        subspaces = self._get_unique_subscapes()
        score_map = self._evaluate_subspaces(subspaces)
        return self._deduce_feature_importances(score_map)

    def _evaluate_subspace(self, features):
        """
        Evaluate a subspace using knn

        Arguments:
            features {df} -- Dataframe containing the features
        """
        features = assert_df(features)
        if self.params["method"] == "imputation" and features.isnull().values.any():
            from fancyimpute import KNN as knn_imputer
            features.update(knn_imputer(k=3, verbose=False).complete(features))

        y = self.data.labels
        cv = StratifiedKFold(y, n_folds=3, shuffle=True)

        clf = knn_classifier(self.data.f_types, self.data.l_type,
                             n_neighbors=self.params["n_neighbors"])

        scoring = "mean_squared_error"
        if self.data.l_type == "nominal":
            scoring = "accuracy"
            y = LabelEncoder().fit_transform(y)
            y = pd.Series(y)

        scores = cross_val_score(clf, features, y, cv=cv, scoring=scoring)
        return np.mean(scores)

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
