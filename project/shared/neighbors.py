"""
    Class for computing distances and nearest neighbors
"""
import numpy as np
from project.utils.assertions import assert_data, assert_series
from scipy.spatial.distance import cdist
from project.shared.c_distance import custom_distance as p_dist


class Neighbors:

    def __init__(self, data, **kwargs):
        """
        Class for calculating distances and nearest neighbors

        Arguments:
            data {data} -- Trainings data
        """
        self.data = assert_data(data)
        self.params = {
            "n_neighbors": kwargs.get("n_neighbors", 3),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def get_nearest_neighbors_fast(self, Y=None):
        D = self.get_dist_matrix(Y)
        N = np.argsort(D)
        N = N[:, :self.params["n_neighbors"]]
        return N, self.data.y

    def get_dist_matrix(self, Y=None):
        kwargs = {
            "nominal_distance": self.params["nominal_distance"],
            "f_types": self.data.f_types.values
        }
        if Y is None:
            D = cdist(self.data.X, self.data.X, metric=p_dist, **kwargs)
        else:
            # TODO: check order
            D = cdist(Y, self.data.X, metric=p_dist, **kwargs)
        return D
