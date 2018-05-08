"""
    Knn Classifier class which uses partial distances
"""
import numpy as np
from collections import Counter
from project import Data
from project.shared.neighbors import Neighbors
from project.utils.assertions import assert_series, assert_data, assert_l_type, assert_df, assert_types


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
            "n_neighbors": kwargs.get("n_neighbors", 3),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def fit(self, X, y):
        """
        Fit the knn classifier

        Arguments:
            X {[type]} -- [description]
            y {[type]} -- [description]
        """
        types = assert_types(self.f_types[X.columns.values], X.columns.values)
        data = Data(X, y, types, self.l_type, X.shape)
        self.Neighbors = Neighbors(data, params=self.params)
        return self

    def predict(self, X):
        """
        Make prediction for unseen samples

        Arguments:
            X {[type]} -- [description]
        """
        X = assert_df(X)
        y_pred = [None] * X.shape[0]
        for row in range(X.shape[0]):
            nn = self.Neighbors.get_nearest_neighbors(X.iloc[row, :])
            # Return most frequent class for nominal labels and mean for numerical
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
