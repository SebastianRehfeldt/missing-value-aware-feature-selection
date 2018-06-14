# %%
import numpy as np

n_samples = 5
n_features = 5
n_independent = 3
n_dependent = 2
n_relevant = 2
y_flip = 0.01
#n_clusters = 2

X = np.zeros((n_samples, n_features))
X

# %%
# Add independent features from N(0,1)
X[:, :n_independent] = np.random.normal(
    loc=0, scale=1.0, size=(n_samples, n_independent))
X

# %%
# Add dependent features
for i in range(n_dependent):
    linear_combination = np.random.normal(0, 1, n_independent)
    print(linear_combination)
    X[:, i + n_independent] = np.sum(
        linear_combination * X[:, :n_independent], axis=1)
X

# %%
# Add random noise
X += np.random.normal(0, 0.01, size=(n_samples, n_features))
X

# %%
# Compute y
# DISCUSS: should give weight to all features but very small values?
relevant_indices = np.random.choice(range(n_independent), n_relevant, False)
weight_vector = np.random.normal(0, 1, n_relevant)
weight_vector /= weight_vector.sum()

# %%
combination = np.sum(X[:, relevant_indices] * weight_vector, axis=1)
y = combination > 0

n_flips = int(np.ceil(n_samples * y_flip))
y_flips = np.random.choice(range(n_samples - 1), n_flips, False)
y[y_flips] = ~y[y_flips]
y
