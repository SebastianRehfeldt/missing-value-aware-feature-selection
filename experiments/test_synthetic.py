# %%
import pandas as pd
from time import time
from pprint import pprint

from experiments.generate_data import create_dataset
from project.utils import introduce_missing_values, scale_data

data, relevance_vector = create_dataset(
    n_samples=1000,
    n_features=20,
    n_independent=20,
    n_dependent=0,
    n_relevant=7,
    n_discrete=0,
    n_clusters=2,
    y_flip=0.01,
    max_features_in_cluster=3,
    max_discrete_values=10)
print(data.shape, flush=True)

data = introduce_missing_values(data, missing_rate=0)
data = scale_data(data)
relevance_vector.sort_values(ascending=False)

# %%
from project.rar.rar import RaR

start = time()
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=4,
    approach="deletion",
    use_pearson=False,
    n_targets=0,
    n_subspaces=5000,
    contrast_iterations=200,
)

rar.fit(data.X, data.y)
ranking = rar.get_ranking()

# %%
CG, i, stop = 0, 0, 5
for feature, score in ranking:
    i += 1
    CG += relevance_vector[feature]

    if i == stop:
        print(CG)
        break
