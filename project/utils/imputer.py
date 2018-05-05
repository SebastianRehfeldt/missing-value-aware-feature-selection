import pandas as pd
from fancyimpute import KNN, MICE, MatrixFactorization, SimpleFill, SoftImpute


class Imputer():

    def __init__(self, data, method="knn"):
        self.data = data
        self.method = method

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return self._complete(pd.DataFrame(X, columns=self.data.features.columns))

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X, y)

    def get_params(self, deep=False):
        return {
            "data": self.data,
            "method": self.method,
        }

    def _complete(self, X):
        # Complete numeric features
        cols = self.data.f_types.loc[self.data.f_types == "numeric"].index
        if X[cols].isnull().values.any():
            X[cols] = self._get_imputer().complete(X[cols])

        # Fancy impute does not handle nominal features
        # TODO: Implement a replace strategy for nominal features using Categorical Encoder
        return X

    def complete(self):
        complete_features = self._complete(self.data.features)
        return self.data._replace(features=complete_features)

    def _get_imputer(self):
        return {
            "knn": KNN,
            "mice": MICE,
            "matrix": MatrixFactorization,
            "simple": SimpleFill,
            "soft": SoftImpute,
        }[self.method]()
