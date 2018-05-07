import sys
from abc import ABC, abstractmethod


class Selector(ABC):
    def __init__(self, data, **kwargs):
        self.data = data
        self.is_fitted = False
        self._init_parameters(kwargs)

    @abstractmethod
    def _init_parameters(self, parameters):
        self.params = {}
        raise NotImplementedError("subclasses must override _init_parameters")

    @abstractmethod
    def calculate_feature_importances(self):
        raise NotImplementedError(
            "subclasses must override calculate_feature_importances")

    def fit(self, X=None, y=None):
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        if not X is None and not y is None:
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            self.data = self.data._replace(features=X, labels=y)

        self.feature_importances = self.calculate_feature_importances()
        self.ranking = self.get_ranking()
        self.selected_features = self.get_selected_features()
        self.is_fitted = True
        return self

    def get_ranking(self):
        return sorted(self.feature_importances.items(),
                      key=lambda k_v: k_v[1], reverse=True)

    def get_selected_features(self):
        return [kv[0] for kv in self.ranking[:self.params["k"]]]

    def transform(self, X=None):
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")
        if not X is None:
            self.data = self.data._replace(features=X)
        return self.data.features[self.selected_features]

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
