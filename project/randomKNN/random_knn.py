import sys
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from fancyimpute import KNN as knn_imputer

from project.randomKNN.knn import KNN as knn_classifier


class RKNN():

    def __init__(self, data, **kwargs):
        self.data = data
        self.is_fitted = False
        self._init_parameters(kwargs)

    def _init_parameters(self, parameters):
        self.params = {
            "n_knn": parameters.get("n_knn", int(self.data.shape[1]**2 / 2)),
            "m": parameters.get("m", int(np.sqrt(self.data.shape[1]))),
            "n_neighbors": parameters.get("n_neighbors", 3),
            "k": parameters.get("k", 3),
            "method": parameters.get("method", "imputation"),
            "nominal_distance": parameters.get("nominal_distance", 1),
        }

    def fit(self, X=None, y=None):
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        if not X is None and not y is None:
            self.data = self.data._replace(features=X, labels=y)
        subspaces = self._get_unique_subscapes()
        score_map = self._evaluate_subspaces(subspaces)
        self.feature_importances = self._deduce_feature_importances(score_map)

        self.ranking = self.get_ranking()
        self.selected_features = [kv[0]
                                  for kv in self.ranking[:self.params["k"]]]
        self.is_fitted = True
        return self

    def _get_unique_subscapes(self):
        subspaces = [list(np.random.choice(self.data.features.columns, self.params["m"], replace=False))
                     for i in range(self.params["n_knn"])]
        subspaces.sort()
        return list(subspaces for subspaces, _ in itertools.groupby(subspaces))

    def _evaluate_subspaces(self, subspaces):
        score_map = defaultdict(list)
        for subspace in subspaces:
            features = self.data.features[subspace]
            mean_score = self._calculate_subspace_score(features)
            for feature in features:
                score_map[feature].append(mean_score)
        return score_map

    def _calculate_subspace_score(self, features):
        if self.params["method"] == "imputation" and features.isnull().values.any():
            features.update(knn_imputer(k=3, verbose=False).complete(features))

        clf = knn_classifier(self.data.f_types, self.data.l_type,
                             n_neighbors=self.params["n_neighbors"])
        y = LabelEncoder().fit_transform(self.data.labels)
        y = pd.Series(y)

        cv = StratifiedKFold(y, n_folds=3, shuffle=True)
        scores = cross_val_score(clf, features, y, cv=cv, scoring="accuracy")
        return np.mean(scores)

    def _deduce_feature_importances(self, score_map):
        return dict((k, np.mean(v)) for k, v in score_map.items())

    def transform(self, X=None):
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")
        if not X is None:
            self.data = self.data._replace(features=X)
        return self.data.features[self.selected_features]

    def get_ranking(self):
        return sorted(self.feature_importances.items(),
                      key=lambda k_v: k_v[1], reverse=True)

    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.data.features.columns.isin(self.selected_features)

    def get_params(self, deep=False):
        return {
            "data": self.data,
            **self.params,
        }
