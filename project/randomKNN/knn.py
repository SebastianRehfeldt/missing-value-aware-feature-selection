import numpy as np
from collections import Counter


class KNN():

    def __init__(self, feature_types, n_neigbors=3):
        self.feature_types = feature_types
        self.n_neigbors = n_neigbors
        self.NOMINAL_DISTANCE = 1

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            classes = self.get_nearest_neighbors(X.iloc[i, :])
            y_pred[i] = Counter(classes).most_common(1)[0][0]
        return y_pred

    def get_nearest_neighbors(self, sample):
        distances = [self.partial_distance(
            sample, self.features.iloc[i, :]) for i in range(self.features.shape[0])]
        indices = np.argsort(distances)[:self.n_neigbors]
        return self.labels[indices]

    def partial_distance(self, x1, x2):
        squared_dist = n_complete = 0
        for i in range(len(x1)):
            is_numerical = self.feature_types[i] == "numeric"

            # only sum up distances between complete pairs
            if is_numerical and not np.isnan(x1[i]) and not np.isnan(x2[i]):
                n_complete += 1
                squared_dist += np.hypot(x1[i], x2[i])

            if not is_numerical and not x1[i] == b'?' and not x2[i] == b'?':
                n_complete += 1
                if x1[i] == x2[i]:
                    squared_dist += self.NOMINAL_DISTANCE

        return np.sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf

    def get_params(self, deep=False):
        return {"n_neigbors": self.n_neigbors, "feature_types": self.feature_types}
