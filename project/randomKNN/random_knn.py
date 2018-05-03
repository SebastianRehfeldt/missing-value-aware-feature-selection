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
        self._init_parameters(kwargs)

    def _init_parameters(self, parameters):
        self.n_samples, self.n_features = self.data.shape
        self.n_knn = parameters.get("n_knn", self.n_features**2)
        self.m = parameters.get("m", int(np.sqrt(self.n_features)))
        self.n_neighbors = parameters.get("n_neighbors", 3)
        self.k = parameters.get("k", 3)
        self.method = parameters.get("method", "imputation")
        self.is_fitted = False

    def fit(self):
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        subspaces = self._get_unique_subscapes()
        score_map = self._evaluate_subspaces(subspaces)
        self.feature_importances = self._deduce_feature_importances(score_map)

        self.ranking = self.get_ranking()
        self.selected_features = [kv[0] for kv in self.ranking[:self.k]]
        self.is_fitted = True
        return self

    def _get_unique_subscapes(self):
        subspaces = [list(np.random.choice(self.data.features.columns, self.m, replace=False))
                     for i in range(self.n_knn)]
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
        if self.method == "imputation" and features.isnull().values.any():
            features.update(knn_imputer(k=3, verbose=False).complete(features))

        clf = knn_classifier(self.data.types, n_neighbors=self.n_neighbors)
        y = LabelEncoder().fit_transform(self.data.labels)
        y = pd.Series(y)

        cv = StratifiedKFold(y, n_folds=3, shuffle=True)
        scores = cross_val_score(clf, features, y, cv=cv, scoring="accuracy")
        return np.mean(scores)

    def _deduce_feature_importances(self, score_map):
        return dict((k, np.mean(v)) for k, v in score_map.items())

    def transform(self):
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")

        return self.data.features[self.selected_features]

    def get_ranking(self):
        return sorted(self.feature_importances.items(),
                      key=lambda k_v: k_v[1], reverse=True)

    def fit_transform(self):
        self.fit()
        return self.transform()

    def get_support(self):
        return self.data.features.columns.isin(self.selected_features)

    def set_params(self, **params):
        pass

    def get_params(self):
        pass
