import numpy as np


class Neighbors:

    def __init__(self, data, **kwargs):
        self.data = data
        self.params = {
            "n_neighbors": kwargs.get("n_neighbors", 3),
            "nominal_distance": kwargs.get("nominal_distance", 1),
        }

    def get_nearest_neighbors(self, sample):
        distances = self.partial_distances(sample)
        indices = np.argsort(distances)[:self.params["n_neighbors"]]
        return self.data.labels.iloc[indices]

    def partial_distances(self, sample):
        return [self.partial_distance(sample, self.data.features.iloc[i, :]) for i in range(self.data.shape[0])]

    def partial_distance(self, x1, x2):
        squared_dist = n_complete = 0

        for i in range(len(x1)):
            is_numerical = self.data.types[i] == "numeric"

            # only sum up distances between complete pairs
            if is_numerical and not np.isnan(x1[i]) and not np.isnan(x2[i]):
                n_complete += 1
                squared_dist += (x1[i] - x2[i])**2

            if not is_numerical:
                n_complete += 1
                if not x1[i] == x2[i]:
                    squared_dist += self.params["nominal_distance"]

        return np.sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf
