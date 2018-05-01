import sys
import itertools
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from fancyimpute import KNN as knn_imputer

from project.randomKNN.knn import KNN as knn_classifier


class RKNN():

    def __init__(self, features, labels, feature_types, **kwargs):
        self.features = features
        self.labels = LabelEncoder().fit_transform(labels)
        self.feature_types = feature_types

        self._init_parameters(kwargs)

    def _init_parameters(self, parameters):
        self.n_samples, self.n_features = self.features.shape
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
        self.is_fitted = True
        return self

    def _get_unique_subscapes(self):
        subspaces = []
        for i in range(self.n_knn):
            features = np.random.choice(
                self.features.columns, self.m, replace=False)
            subspaces.append(list(features))

        subspaces.sort()
        subspaces = list(subspaces for subspaces,
                         _ in itertools.groupby(subspaces))
        return subspaces

    def _evaluate_subspaces(self, subspaces):
        score_map = defaultdict(list)
        for subspace in subspaces:
            features = self.features[subspace]
            mean_score = self._calculate_subspace_score(features)
            for feature in features:
                score_map[feature].append(mean_score)
        return score_map

    def _calculate_subspace_score(self, features):
        if self.method == "imputation":
            if features.isnull().values.any():
                features = knn_imputer(k=3, verbose=False).complete(features)
            knn_clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        else:
            knn_clf = knn_classifier(
                self.feature_types, n_neigbors=self.n_neighbors)
        scores = cross_val_score(
            knn_clf, features, self.labels, cv=5, scoring="accuracy")
        return np.mean(scores)

    def _deduce_feature_importances(self, score_map):
        feature_importances = defaultdict(float)
        for key, values in score_map.items():
            feature_importances[key] = np.mean(values)

        return feature_importances

    def transform(self):
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")

        ranking = self.get_ranking()
        self.selected_features = list(ranking.keys())[:self.k]
        return self.features[self.selected_features]

    def get_ranking(self):
        ranking = sorted(self.feature_importances.items(),
                         key=lambda k_v: k_v[1], reverse=True)
        return dict(ranking)

    def fit_transform(self):
        self.fit()
        return self.transform()

    def get_support(self):
        return self.features.columns.isin(self.selected_features)

    def set_params(self, **params):
        pass

    def get_params(self):
        pass
