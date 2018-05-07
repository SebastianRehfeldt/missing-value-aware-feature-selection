"""
    Wrapper class for fancy impute to match sklearn requirements
"""
import pandas as pd
from fancyimpute import KNN, MICE, MatrixFactorization, SimpleFill, SoftImpute
from project.utils.assertions import assert_data


class Imputer():

    def __init__(self, data, method="knn"):
        """
        Imputer class for filling missing values

        Arguments:
            data {Data} -- Data object which should be filled

        Keyword Arguments:
            method {str} -- Imputation technique (default: {"knn"})
        """
        self.data = assert_data(data)
        self.method = method

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
        return self._complete(pd.DataFrame(X, columns=self.data.features.columns))

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
            "data": self.data,
            "method": self.method,
        }

    def _complete(self, X):
        """
        Complete numeric features

        Arguments:
            X {df} -- Feature matrix which should be filled
        """
        cols = self.data.f_types.loc[self.data.f_types == "numeric"].index
        if X[cols].isnull().values.any():
            X[cols] = self._get_imputer().complete(X[cols])

        # Fancy impute does not handle nominal features
        # TODO: Implement a replace strategy for nominal features using Categorical Encoder
        return X

    def complete(self):
        """
        Complete features
        """
        complete_features = self._complete(self.data.features)
        complete_data = self.data._replace(features=complete_features)
        complete_data = assert_data(complete_data)
        return complete_data

    def _get_imputer(self):
        """
        Get imputer for method
        """
        return {
            "knn": KNN,
            "mice": MICE,
            "matrix": MatrixFactorization,
            "simple": SimpleFill,
            "soft": SoftImpute,
        }[self.method]()
