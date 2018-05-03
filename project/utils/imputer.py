import pandas as pd
from fancyimpute import KNN, MICE, MatrixFactorization, SimpleFill, SoftImpute


class Imputer():

    def __init__(self, data, method="knn"):
        self.data = data
        self.method = method

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        imputer = self._get_imputer()
        complete_features = imputer.complete(X)
        return pd.DataFrame(complete_features, columns=self.data.features.columns)

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X, y)

    def get_params(self, deep=False):
        return {
            "data": self.data,
            "method": self.method,
        }

    def complete(self):
        complete_features = self._get_imputer().complete(self.data.features)
        return self.data._replace(features=complete_features)

    def _get_imputer(self):
        return {
            "knn": KNN,
            "mice": MICE,
            "matrix": MatrixFactorization,
            "simple": SimpleFill,
            "soft": SoftImpute,
        }[self.method]()
