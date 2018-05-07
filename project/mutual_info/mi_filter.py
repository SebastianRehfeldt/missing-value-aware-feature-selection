import sys
import numpy as np
import pandas as pd
from project.shared.neighbors import Neighbors
from project.shared.mutual_information import get_mutual_information


class MI_Filter():
    def __init__(self, data, **kwargs):
        self.data = data
        self.is_fitted = False
        self._init_parameters(kwargs)

    def _init_parameters(self, parameters):
        self.params = {
            "k": parameters.get("k", 3),
            "nominal_distance": parameters.get("nominal_distance", 1),
        }

    def fit(self, X=None, y=None):
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        if not X is None and not y is None:
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            self.data = self.data._replace(features=X, labels=y)

        self.feature_importances = self.get_individual_mis()

        self.ranking = self.get_ranking()
        self.selected_features = [kv[0]
                                  for kv in self.ranking[:self.params["k"]]]
        self.is_fitted = True
        return self

    def get_individual_mis(self):
        scores = {}
        for col in self.data.features:
            features = self.data.features[col].to_frame()
            types = pd.Series(self.data.f_types[col], [col])

            selected_data = self.data._replace(
                features=features, f_types=types, shape=features.shape)

            scores[col] = get_mutual_information(selected_data)
        return scores

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
