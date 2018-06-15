# %%
import numpy as np
import pandas as pd
from .data import Data

# TODO: Discuss
# Shuffle columns necessary?
# Discretization lowers relevance
# Relevance of features in clusters (inside multiple clusters?)
# Relevance of dependent features


class DataGenerator():
    def __init__(self, **params):
        self._init_params(**params)

        self.X = np.zeros((self.n_samples, self.n_features + self.n_clusters))
        self.data = None

    def _init_params(self, **params):
        self.n_samples = params.get("n_samples", 1000)
        self.n_features = params.get("n_features", 20)
        self.n_independent = params.get("n_independent", 18)
        self.n_dependent = params.get("n_dependent", 2)
        self.n_relevant = params.get("n_relevant", 3)
        self.n_discrete = params.get("n_discrete", 0)
        self.n_clusters = params.get("n_clusters", 0)
        self.y_flip = params.get("y_flip", 0.01)
        self.max_features_in_cluster = params.get("max_features_in_cluster", 3)
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

            pos = self.n_clusters - i
            signs = self.xor_signs(subset)
            values = self.xor_values(subset)
            self.X[:, -pos] = signs * values

    def _add_dependent(self):
        self.linear_combination = [None] * self.n_dependent
        for i in range(self.n_dependent):
            linear_combination = np.random.uniform(0, 1, self.n_independent)
            self.linear_combination[i] = linear_combination

            self.X[:, i + self.n_independent] = np.sum(
                linear_combination * self.X[:, :self.n_independent], axis=1)

    def _create_relevance_vector(self):
        self.relevance_features = np.random.uniform(0, 1, self.n_relevant)
        self.relevance_clusters = np.random.uniform(0, 1, self.n_clusters)

        relevance_vector = np.zeros(self.n_features)
        relevance_vector[:self.n_relevant] = self.relevance_features

        for i in range(self.n_clusters):
            subset = self.clusters[i]
            rel = self.relevance_clusters[i] / len(subset)
            for idx in subset:
                if relevance_vector[idx] == 0:
                    relevance_vector[idx] = rel
                else:
                    relevance_vector[idx] = (relevance_vector[idx] + rel) / 2
        relevance_vector /= np.sum(relevance_vector)
        self.relevance_vector = relevance_vector

    def _compute_label_vector(self):
        combination = np.sum(
            self.X[:, :self.n_relevant] * self.relevance_features, axis=1)

        if self.n_clusters > 0:
            combination_c = np.sum(
                self.X[:, -self.n_clusters:] * self.relevance_clusters, axis=1)
            combination += combination_c

        # Compute target vector by using sign and flipping
        y = combination > 0

        n_flips = int(np.ceil(self.n_samples * self.y_flip))
        y_flips = np.random.choice(range(self.n_samples - 1), n_flips, False)
        y[y_flips] = ~y[y_flips]

        self.y = y.astype(int).astype(str)

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
        names = ["feature" + str(i) for i in range(self.n_features)]

        X = pd.DataFrame(self.X[:, :self.n_features], columns=names)
        y = pd.Series(self.y, name="class")

        f_types = pd.Series(["numeric"] * self.n_features, index=names)
        f_types[self.discrete_features] = "nominal"
        l_type = "nominal"

        self.data = Data(X, y, f_types, l_type, X.shape)
        self.relevance_vector = pd.Series(self.relevance_vector, index=names)

    def create_dataset(self):
        # Add independent features from N(0,1)
        self.X[:, :self.n_independent] = np.random.normal(
            0, 1.0, (self.n_samples, self.n_independent))

        # Add clusters on irrelevant features to last columns of array
        self._add_clusters()

        # Add dependent features as linear combination of irrelevant features
        self._add_dependent()

        # Add random noise
        self.X += np.random.normal(0, 0.1, size=self.X.shape)

        # Create relevance vector
        self._create_relevance_vector()

        # Compute linear combination and set labels
        self._compute_label_vector()

        # Discretize features based on hidden state of feature
        self._discretize_features()

        # Create data object and add names to relevance vector
        self._finalize()
        return self.data, self.relevance_vector

    def get_dataset(self):
        if self.data is None:
            self.create_dataset()

        return self.data, self.relevance_vector
