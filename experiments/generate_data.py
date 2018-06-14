# %%
import numpy as np


def xor_signs(X, subspace):
    signs = (X[:, subspace] >= 0).astype(int)
    signs = np.sum(signs, axis=1) % 2 * 2 - 1
    return signs


def xor_values(X, subspace):
    sums = np.sum(np.log(np.abs(X[:, subspace])), axis=1)
    values = np.exp(sums / len(subspace))
    return values


def create_dataset(n_samples=5,
                   n_features=20,
                   n_independent=15,
                   n_dependent=5,
                   n_relevant=3,
                   n_discrete=1,
                   n_clusters=2,
                   y_flip=0.01,
                   max_features_in_cluster=3,
                   max_discrete_values=10):

    X = np.zeros((n_samples, n_features + n_clusters))

    # Add independent features from N(0,1)
    X[:, :n_independent] = np.random.normal(0, 1.0, (n_samples, n_independent))

    # Add clusters on irrelevant features to last columns of array
    relevant_indices = set([i for i in range(n_relevant)])
    clusters = {}
    for i in range(n_clusters):
        indices = [
            idx + n_relevant for idx in range(n_independent - n_relevant)
        ]
        n_features_in_clust = np.random.randint(2, max_features_in_cluster + 1)
        subset = np.random.choice(indices, n_features_in_clust, False)
        relevant_indices |= set(subset)
        clusters[i] = subset

        signs = xor_signs(X, subset)
        values = xor_values(X, subset)
        pos = n_clusters - i
        X[:, -pos] = signs * values

    # Add dependent features
    relevant_indices = np.asarray(list(relevant_indices))
    for i in range(n_dependent):
        lc = np.random.uniform(0, 1, len(relevant_indices))
        X[:, i + n_independent] = np.sum(lc * X[:, relevant_indices], axis=1)

    # Add random noise
    X += np.random.normal(0, 0.1, size=(n_samples, n_features + n_clusters))

    # Compute class vector
    # TODO: relevance vector should contain all features
    # TODO: distribute relevance of cluster to contained features
    # TODO: normalization of weights
    relevance_vector_features = np.random.uniform(0, 1, n_relevant)
    relevance_vector_clusters = np.random.uniform(0, 1, n_clusters)

    combination_features = np.sum(
        X[:, :n_relevant] * relevance_vector_features, axis=1)
    combination_clusters = np.sum(
        X[:, -n_clusters:] * relevance_vector_clusters, axis=1)

    combination = combination_features + combination_clusters

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

    # Create data object and remove clusters, shuffle, provide relevance vector
    return X, y


X, y = create_dataset()
X

# %%
# TODO: Discuss
# LC Weights equal relevance?
# Discretization lowers relevance
# Relevance of features in clusters
