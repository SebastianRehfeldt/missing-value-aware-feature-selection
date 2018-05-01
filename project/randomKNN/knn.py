import numpy as np
from scipy.stats import mode


class KNN():

    def __init__(self, types=[], n_neigbors=3):
        self.types = types
        self.n_neigbors = n_neigbors

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            classes = self.get_nearest_neighbors(X.iloc[i, :])
            y_pred[i] = mode(classes).mode[0]
        return y_pred

    def get_nearest_neighbors(self, sample):
        distances = [self.partial_distance(
            sample, self.features.iloc[i, :]) for i in range(self.features.shape[0])]
        indices = np.argsort(distances)[:self.n_neigbors]
        return self.labels[indices]

    def partial_distance(self, x1, x2):
        squared_dist = n_complete = 0
        for i in range(len(x1)):
            # only sum up distances between complete pairs
            if not np.isnan(x1[i]) and not np.isnan(x2[i]):
                n_complete += 1
                squared_dist += np.hypot(x1[i], x2[i])
        return np.sqrt(squared_dist / n_complete) if n_complete > 0 else np.inf

    def get_params(self, deep=False):
        return {"n_neigbors": self.n_neigbors, "types": self.types}
