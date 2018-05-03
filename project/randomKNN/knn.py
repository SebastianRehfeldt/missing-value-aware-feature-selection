import numpy as np
from collections import Counter
from project import Data
from project.shared.neighbors import Neighbors


class KNN():

    def __init__(self, types, **kwargs):
        self.types = types
        self.params = {
            "n_neighbors": kwargs.get("n_neighbors", 3),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def fit(self, features, labels):
        self.data = Data(features, labels, self.types, features.shape)
        self.Neighbors = Neighbors(self.data, params=self.params)
        return self

    def predict(self, X):
        y_pred = [None] * X.shape[0]
        for i in range(X.shape[0]):
            classes = self.Neighbors.get_nearest_neighbors(X.iloc[i, :])
            y_pred[i] = Counter(classes).most_common(1)[0][0]
        return y_pred

    def get_params(self, deep=False):
        return {
            "types": self.types,
            **self.params,
        }
