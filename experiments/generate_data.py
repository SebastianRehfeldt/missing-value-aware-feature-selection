# %%
import numpy as np
import pandas as pd
from project.utils import Data

# TODO: Discuss
# LC Weights equal relevance?
# Discretization lowers relevance
# Relevance of features in clusters (inside multiple clusters?)
# Relevance of dependent features


def xor_signs(X, subspace):
    signs = (X[:, subspace] >= 0).astype(int)
    signs = np.sum(signs, axis=1) % 2 * 2 - 1
    return signs


def xor_values(X, subspace):
    sums = np.sum(np.log(np.abs(X[:, subspace])), axis=1)
    values = np.exp(sums / len(subspace))
    return values


def create_dataset(n_samples=1000,
                   n_features=20,
                   n_independent=18,
                   n_dependent=2,
                   n_relevant=3,
                   n_discrete=0,
                   n_clusters=0,
                   y_flip=0.01,
                   max_features_in_cluster=3,
                   max_discrete_values=10):

    X = np.zeros((n_samples, n_features + n_clusters))

    # Add independent features from N(0,1)
    X[:, :n_independent] = np.random.normal(0, 1.0, (n_samples, n_independent))

    # Add clusters on irrelevant features to last columns of array
    clusters = {}
    for i in range(n_clusters):
        indices = [
            idx + n_relevant for idx in range(n_independent - n_relevant)
        ]
        n_features_in_clust = np.random.randint(2, max_features_in_cluster + 1)
        subset = np.random.choice(indices, n_features_in_clust, False)
        clusters[i] = subset

        signs = xor_signs(X, subset)
        values = xor_values(X, subset)
        pos = n_clusters - i
        X[:, -pos] = signs * values

    # Add dependent features
    for i in range(n_dependent):
        lc = np.random.uniform(0, 1, n_independent)
        X[:, i + n_independent] = np.sum(lc * X[:, :n_independent], axis=1)

    # Add random noise
    X += np.random.normal(0, 0.1, size=(n_samples, n_features + n_clusters))

    # Compute relevance vectors
    relevance_features = np.random.uniform(0, 1, n_relevant)
    relevance_clusters = np.random.uniform(0, 1, n_clusters)

    relevance_vector = np.zeros(n_features)
    relevance_vector[:n_relevant] = relevance_features
    for i in range(n_clusters):
        rel = relevance_clusters[i] / len(clusters[i])
        for idx in clusters[i]:
            if relevance_vector[idx] == 0:
                relevance_vector[idx] = rel
            else:
                relevance_vector[idx] = (relevance_vector[idx] + rel) / 2
    relevance_vector /= np.sum(relevance_vector)

    # Compute linear combination
    combination = np.sum(X[:, :n_relevant] * relevance_features, axis=1)
    if n_clusters > 0:
        combination_c = np.sum(X[:, -n_clusters:] * relevance_clusters, axis=1)
        combination += combination_c

    # Compute target vector by using sign and flipping
    y = combination > 0
    n_flips = int(np.ceil(n_samples * y_flip))
    y_flips = np.random.choice(range(n_samples - 1), n_flips, False)
    y[y_flips] = ~y[y_flips]
    y = y.astype(int).astype(str)

    # Discretize features
    discrete_features = np.random.choice(range(n_features), n_discrete, False)
    for index in discrete_features:
        n_values = np.random.randint(2, max_discrete_values)
        values = list(range(n_values))
        values_pos = [val for val in values if np.random.random() >= 0.5]
        values_neg = [val for val in values if val not in values_pos]

        if len(values_pos) == 0:
            values_pos = [values_neg[0]]
            del values_neg[0]
        if len(values_neg) == 0:
            values_neg = [values_pos[0]]
            del values_pos[0]

        idx_pos = np.where(X[:, index] >= 0)[0]
        idx_neg = np.where(X[:, index] < 0)[0]

        X[idx_pos, index] = np.random.choice(values_pos, len(idx_pos))
        X[idx_neg, index] = np.random.choice(values_neg, len(idx_neg))

    names = ["feature" + str(i) for i in range(n_features)]
    f_types = pd.Series(["numeric"] * n_features, index=names)
    f_types[discrete_features] = "nominal"
    l_type = "nominal"

    X = pd.DataFrame(X[:, :n_features], columns=names)
    y = pd.Series(y, name="class")

    data = Data(X, y, f_types, l_type, X.shape)
    relevance_vector = pd.Series(relevance_vector, index=names)
    # Create data object and remove clusters, shuffle, provide relevance vector
    return data, relevance_vector


if __name__ == '__main__':
    data, relevance_vector = create_dataset()
    print(data.X)
