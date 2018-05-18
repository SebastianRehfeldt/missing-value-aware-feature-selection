"""
    Knn Classifier class which uses partial distances
"""
import numpy as np
from collections import Counter

from project.utils import assert_series, assert_l_type, assert_df, assert_types
from project.shared import get_dist_matrix


class KNN():
    def __init__(self, f_types, l_type, **kwargs):
        """
        Class which predicts label for unseen samples

        Arguments:
            f_types {pd.series} -- Series of feature types
            l_type {str} -- Type of label
        """
        self.f_types = assert_series(f_types)
        self.l_type = assert_l_type(l_type)
        self.params = {
            "n_neighbors": kwargs.get("n_neighbors", 6),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def fit(self, X, y):
        """
        Fit the knn classifier

        Arguments:
            X {[df]} -- Dataframe containing the features
            y {pd.series} -- Label vector
        """
        types = assert_types(self.f_types[X.columns.values], X.columns.values)
        self.f_types = types
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X_test):
        """
        Make prediction for unseen samples

        Arguments:
            X {[df]} -- Dataframe containing the features
        """
        X_test = assert_df(X_test)
        N = self.get_nearest_neighbors(X_test)

        y_pred = [None] * X_test.shape[0]
        for row in range(X_test.shape[0]):
            nn = self.y_train.iloc[N[row, :]]
            if self.l_type == "nominal":
                y_pred[row] = Counter(nn).most_common(1)[0][0]
            else:
                y_pred[row] = np.mean(nn)
        return y_pred

    def get_params(self, deep=False):
        """
        Return params

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "f_types": self.f_types,
            "l_type": self.l_type,
            **self.params,
        }

    def get_nearest_neighbors(self, X_test):
        D = get_dist_matrix(self.X_train, self.f_types, X_test, **self.params)
        N = np.argsort(D)
        return N[:, :self.params["n_neighbors"]]
