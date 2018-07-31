"""
    Wrapper class for fancy impute to match sklearn requirements
"""
import pandas as pd

from fancyimpute import KNN, MICE, MatrixFactorization, SimpleFill, SoftImpute
from project.utils import assert_data


class Imputer():
    def __init__(self, f_types, strategy="knn"):
        """
        Imputer class for filling missing values

        Arguments:
            f_types {pd.Series} -- Series containing the feature types
        Keyword Arguments:
            strategy {str} -- Imputation technique (default: {"knn"})
        """
        self.f_types = f_types
        self.strategy = strategy

    def fit(self, X=None, y=None):
        """
        Fit method which is neccessary for sklearn, data is set in init

        Keyword Arguments:
            X {df} -- Feature matrix (default: {None})
            y {pd.series} -- Label vector (default: {None})
        """
        return self

    def transform(self, X, y=None):
        """
        Transform features by filling empty spots

        Arguments:
            X {df} -- Feature Matrix

        Keyword Arguments:
            y {pd.series} -- Label vector (default: {None})
        """
        self.f_types = self.f_types[X.columns]
        return self._complete(
            pd.DataFrame(X, columns=self.f_types.index.tolist()))

    def fit_transform(self, X, y=None):
        """
        Fit imputer and complete data

        Arguments:
            X {df} -- Feature Matrix

        Keyword Arguments:
            y {pd.series} -- Label vector (default: {None})
        """
        self.fit()
        return self.transform(X, y)

    def get_params(self, deep=False):
        """
        Return params needed to copy objects in sklearn

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "f_types": self.f_types,
            "strategy": self.strategy,
        }

    def _complete(self, X):
        """
        Complete numeric features

        Arguments:
            X {df} -- Feature matrix which should be filled
        """
        cols = self.f_types.loc[self.f_types == "numeric"].index
        X_copy = X.copy()
        if X[cols].isnull().values.any():
            X_copy[cols] = self._get_imputer().complete(X_copy[cols])

        for col in X_copy:
            if self.f_types[col] == "nominal":
                elements = X_copy[col].value_counts()
                mode = elements.index[0]
                if mode == "?":
                    mode = elements.index[1]

                indices = X_copy[col] == "?"
                X_copy[col][indices] = mode

        return X_copy

    def complete(self, data):
        """
        Complete features
        """
        data = assert_data(data)
        complete_features = self._complete(data.X)
        return data.replace(copy=True, X=complete_features)

    def _get_imputer(self):
        """
        Get imputer for strategy
        """
        if self.strategy == "simple":
            return SimpleFill()

        return {
            "knn": KNN,
            "mice": MICE,
            "matrix": MatrixFactorization,
            "soft": SoftImpute,
        }[self.strategy](verbose=False)
