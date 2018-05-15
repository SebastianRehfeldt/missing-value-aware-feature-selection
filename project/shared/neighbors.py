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
            D = cdist(Y, self.data.X, metric=p_dist, **kwargs)
        return D

    def get_nearest_neighbors(self, sample):
        """
        Return labels of nearest neighbors for a sample

        Arguments:
            sample {pd.series} -- Sample which is compared to trainings data
        """
        distances = self.partial_distances(sample)
        k = min(self.params["n_neighbors"], len(distances))
        indices = np.argsort(distances)[:k]
        return self.data.y.iloc[indices]

    def partial_distances(self, sample):
        """
        Returns partial distances of a sample compared to trainings data

        Arguments:
            sample {pd.series} -- Sample which is compared to trainings data
        """
        sample = assert_series(sample)
        return [self.partial_distance(sample, self.data.X.iloc[row, :]) for row in range(self.data.shape[0])]

    def partial_distance(self, x1, x2):
        """
        Returns partial distance between two samples

        Arguments:
            x1 {pd.series} -- Sample
            x2 {pd.series} -- Sample
        """
        # TODO check that f_type matches the right feature
        # print(x1, self.data.f_types)
        x1 = assert_series(x1)
        x2 = assert_series(x2)
        assert (len(x1) == len(x2)), "samples have different lengths"

        squared_dist = n_complete = 0
        for i in range(len(x1)):
            is_numerical = self.data.f_types[i] == "numeric"

            # Only sum up distances between complete pairs
            if is_numerical and not np.isnan(x1[i]) and not np.isnan(x2[i]):
                n_complete += 1
                squared_dist += (x1[i] - x2[i])**2

            if not is_numerical:
                n_complete += 1
                if not x1[i] == x2[i]:
                    squared_dist += self.params["nominal_distance"]

        return np.sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf
