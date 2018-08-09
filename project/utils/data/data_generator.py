import numpy as np
import pandas as pd
from .data import Data

# TODO: Discuss
# Discretization lowers relevance
# Relevance of features in clusters (inside multiple clusters?)
# Relevance of dependent features


class DataGenerator():
    def __init__(self, random_state=None, **params):
        self.set_seed(random_state)
        self._init_params(**params)

        n = self.n_features + self.n_clusters + self.n_informative_missing
        self.X = np.zeros((self.n_samples, n))
        self.data = None

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def get_clusters(self):
        clusters = self.clusters.copy()
        return [v for k, v in clusters.items()]

    def _init_params(self, **params):
        self.n_samples = params.get("n_samples", 500)
        self.n_features = params.get("n_features", 20)
        self.n_independent = params.get("n_independent", 20)
        self.n_dependent = params.get("n_dependent", 0)
        self.n_relevant = params.get("n_relevant", 0)
        self.n_informative_missing = params.get("n_informative_missing", 0)
        self.missing_rate = params.get("missing_rate", 0.3)
        self.n_discrete = params.get("n_discrete", 0)
        self.n_clusters = params.get("n_clusters", 3)
        self.y_flip = params.get("y_flip", 0.01)
        self.max_features_in_cluster = params.get("max_features_in_cluster", 2)
        self.max_discrete_values = params.get("max_discrete_values", 10)

    def xor_signs(self, subspace):
        signs = (self.X[:, subspace] >= 0).astype(int)
        signs = np.sum(signs, axis=1) % 2 * 2 - 1
        return signs

    def xor_values(self, subspace):
        sums = np.sum(np.log(np.abs(self.X[:, subspace])), axis=1)
        values = np.exp(sums / len(subspace))
        return values

    def _add_clusters(self):
        self.clusters = {}
        for i in range(self.n_clusters):
            indices = [
                idx + self.n_relevant
                for idx in range(self.n_independent - self.n_relevant)
            ]
            n_features_in_clust = np.random.randint(
                2, self.max_features_in_cluster + 1)
            subset = np.random.choice(indices, n_features_in_clust, False)
            self.clusters[i] = subset

            pos = self.n_clusters - i + self.n_informative_missing
            signs = self.xor_signs(subset)
            values = self.xor_values(subset)
            self.X[:, -pos] = signs * values

    def _add_dependent(self):
        self.linear_combination = [None] * self.n_dependent
        for i in range(self.n_dependent):
            linear_combination = np.random.uniform(0.2, 1, self.n_independent)
            self.linear_combination[i] = linear_combination

            self.X[:, i + self.n_independent] = np.sum(
                linear_combination * self.X[:, :self.n_independent], axis=1)

    def _add_informative_missing(self):
        for i in range(self.n_informative_missing):
            pos = self.n_informative_missing - i
            n = int(self.n_samples * self.missing_rate)
            idx = np.random.choice(range(self.n_samples), n, False)
            self.X[idx, -pos] = 1

    def _create_relevance_vector(self):
        self.relevance_features = np.random.uniform(0.2, 1, self.n_relevant)
        self.relevance_clusters = np.random.uniform(0.2, 1, self.n_clusters)
        self.relevance_missing = np.random.uniform(0.2, 1,
                                                   self.n_informative_missing)

        relevance_vector = np.zeros(self.n_features)
        relevance_vector[:self.n_relevant] = self.relevance_features

        for i in range(self.n_clusters):
            subset = self.clusters[i]
            rel = self.relevance_clusters[i] / len(subset)
            for idx in subset:
                if relevance_vector[idx] == 0:
                    relevance_vector[idx] = rel
                else:
                    relevance_vector[idx] = relevance_vector[idx] + rel

        if self.n_informative_missing > 0:
            irrelevant = np.where(relevance_vector == 0)[0]
            m = np.random.choice(irrelevant, self.n_informative_missing, False)
            self.informative_missing = m
            for idx in range(self.n_informative_missing):
                score = self.relevance_missing[idx] * self.missing_rate
                relevance_vector[m[idx]] = score

        relevance_vector /= np.sum(relevance_vector)
        self.relevance_vector = relevance_vector

    def _compute_label_vector(self):
        combination = np.sum(
            self.X[:, :self.n_relevant] * self.relevance_features, axis=1)

        if self.n_clusters > 0:
            s = -self.n_clusters - self.n_informative_missing
            e = -self.n_informative_missing
            combination_c = np.sum(
                self.X[:, s:e] * self.relevance_clusters, axis=1)
            combination += combination_c

        if self.n_informative_missing > 0:
            pos = -self.n_informative_missing
            combination_m = np.sum(
                self.X[:, pos:] * self.relevance_missing, axis=1)
            combination += combination_m

        # Compute target vector by using sign and flipping
        y = combination > 0

        n_flips = int(np.ceil(self.n_samples * self.y_flip))
        y_flips = np.random.choice(range(self.n_samples - 1), n_flips, False)
        y[y_flips] = ~y[y_flips]

        self.y = y.astype(int).astype(str)

    def _remove_informative_missing(self):
        for i in range(self.n_informative_missing):
            dummy = self.X[:, -self.n_informative_missing + i]
            idx = np.where(dummy == 1)[0]
            target = self.informative_missing[i]
            self.X[idx, target] = np.nan

    def _discretize_features(self):
        self.discrete_features = np.random.choice(
            range(self.n_features), self.n_discrete, False)

        for index in self.discrete_features:
            n_values = np.random.randint(2, self.max_discrete_values)
            values = list(range(n_values))

            values_pos = [val for val in values if np.random.random() >= 0.5]
            values_neg = [val for val in values if val not in values_pos]

            # Assure that values_pos and values_neg contain at least 1 value
            if len(values_pos) == 0:
                values_pos = [values_neg[0]]
                del values_neg[0]
            if len(values_neg) == 0:
                values_neg = [values_pos[0]]
                del values_pos[0]

            idx_pos = np.where(self.X[:, index] >= 0)[0]
            idx_neg = np.where(self.X[:, index] < 0)[0]

            self.X[idx_pos, index] = np.random.choice(values_pos, len(idx_pos))
            self.X[idx_neg, index] = np.random.choice(values_neg, len(idx_neg))

    def _finalize(self):
        names = ["f" + str(i) for i in range(self.n_features)]

        X = pd.DataFrame(self.X[:, :self.n_features], columns=names)
        y = pd.Series(self.y, name="class")

        f_types = pd.Series(["numeric"] * self.n_features, index=names)
        f_types[self.discrete_features] = "nominal"
        l_type = "nominal"

        shuffled_names = np.random.permutation(names)
        X = X[shuffled_names]
        f_types = f_types[shuffled_names]

        self.data = Data(X, y, f_types, l_type, X.shape)
        self.relevance_vector = pd.Series(self.relevance_vector, index=names)
        self.relevance_vector = self.relevance_vector[shuffled_names]

    def create_dataset(self):
        # Add independent features from N(0,1)
        self.X[:, :self.n_independent] = np.random.normal(
            0, 1.0, (self.n_samples, self.n_independent))

        # Add clusters on irrelevant features to last columns of array
        self._add_clusters()

        # Add dependent features as linear combination of irrelevant features
        self._add_dependent()

        # TODO
        self._add_informative_missing()

        # Add random noise
        size = (self.X.shape[0], self.X.shape[1] - self.n_informative_missing)
        self.X[:, :-self.n_informative_missing] += np.random.normal(
            0, 0.1, size=size)

        # Create relevance vector
        self._create_relevance_vector()

        # Compute linear combination and set labels
        self._compute_label_vector()

        # TODO
        self._remove_informative_missing()

        # Discretize features based on hidden state of feature
        self._discretize_features()

        # Create data object and add names to relevance vector
        self._finalize()
        return self.data, self.relevance_vector

    def get_dataset(self):
        if self.data is None:
            self.create_dataset()

        return self.data, self.relevance_vector

    def get_params(self):
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_independent": self.n_independent,
            "n_dependent": self.n_dependent,
            "n_relevant": self.n_relevant,
            "n_discrete": self.n_discrete,
            "n_clusters": self.n_clusters,
            "y_flip": self.y_flip,
            "max_features_in_cluster": self.max_features_in_cluster,
            "max_discrete_values": self.max_discrete_values,
            "seed": self.seed,
        }
