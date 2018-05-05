import numpy as np
from collections import Counter
from project import Data
from project.shared.neighbors import Neighbors


class KNN():

    def __init__(self, f_types, l_type, **kwargs):
        self.f_types = f_types
        self.l_type = l_type
        self.params = {
            "n_neighbors": kwargs.get("n_neighbors", 3),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def fit(self, X, y):
        data = Data(X, y, self.f_types, self.l_type, X.shape)
        self.Neighbors = Neighbors(data, params=self.params)
        return self

    def predict(self, X):
        y_pred = [None] * X.shape[0]
        for row in range(X.shape[0]):
            classes = self.Neighbors.get_nearest_neighbors(X.iloc[row, :])
            y_pred[row] = Counter(classes).most_common(1)[0][0]
        return y_pred

    def get_params(self, deep=False):
        return {
            "f_types": self.f_types,
            "l_type": self.l_type,
            **self.params,
        }
