# %%
import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from project import EXPERIMENTS_PATH
from project.rar.rar import RaR
from project.classifier import KNN
from project.utils import DataLoader, scale_data
from project.utils.data import DataGenerator

is_real_data = True
use_redundancy = False

if is_real_data:
    name = "ionosphere"
    data_loader = DataLoader(ignored_attributes=["molecule_name"])
    data = data_loader.load_data(name, "arff")
    thresh = int(np.sqrt(data.shape[1])) + 1
    k = [1, 2, 3, 5, 8, 10, 15, 20]
    k = [x for x in k if x <= thresh]
else:
    dataset_params = {}
    generator = DataGenerator(42, **dataset_params)
    data, relevances = generator.create_dataset()
    name = "_".join(map(str, list(generator.get_params().values())))
    k = [int((relevances > 0).sum())]

data = scale_data(data)
data.shuffle_rows()

FOLDER = os.path.join(EXPERIMENTS_PATH, "grid", name)
os.makedirs(FOLDER, exist_ok=True)

# INIT RAR AND KNN
knn = KNN(data.f_types, data.l_type, knn_neighbors=6)
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
)
pipe = Pipeline([('reduce_dim', rar), ('classify', knn)])

# PARAM GRID
n_targets = 3 if use_redundancy else 0
min_subspaces = min(600, data.shape[1] * 3)
max_subspaces = max(100, data.shape[1]**2)
n_subspaces = [50, 100, 200, 400, 600, 900, 1200, 1600, 2000]
n_subspaces = [n for n in n_subspaces if min_subspaces <= n <= max_subspaces]
param_grid = [
    {
        'reduce_dim': [RaR(data.f_types, data.l_type, data.shape)],
        'reduce_dim__k': k,
        'reduce_dim__approach': ['deletion'],
        'reduce_dim__n_targets': n_targets,
        'reduce_dim__contrast_iterations': [100, 250],
        'reduce_dim__alpha': [0.1, 0.05, 0.02, 0.01],
        'reduce_dim__n_subspaces': n_subspaces,
    },
]

# FIT GRID SEARCH
grid = GridSearchCV(
    pipe,
    cv=3,
    n_jobs=4,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, average="micro"),
)
grid.fit(data.X, data.y)

# STORE RESULTS AND RANKING
results = pd.DataFrame(grid.cv_results_)
results.sort_values(by="mean_test_score", ascending=False, inplace=True)
results.to_csv(os.path.join(FOLDER, "results.csv"))

ranking = grid.best_estimator_.steps[0][1].get_ranking()
ranking = dict(ranking)
ranking = pd.DataFrame(
    data={"score": list(ranking.values())},
    index=ranking.keys(),
)
ranking.to_csv(os.path.join(FOLDER, "ranking.csv"))

# %%
# STORE BEST PARAMS
with open(os.path.join(FOLDER, "config.json"), 'w') as filepath:
    best_params = results.iloc[0].params["reduce_dim"].get_params()
    del best_params["f_types"]
    del best_params["l_type"]
    del best_params["shape"]
    json.dump(best_params, filepath, indent=4)

# STORE PARAMS GRID
with open(os.path.join(FOLDER, "param_grid.json"), 'w') as filepath:
    params = param_grid.copy()[0]
    del params["reduce_dim"]
    json.dump(param_grid, filepath, indent=4)

if not is_real_data:
    relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
    with open(os.path.join(FOLDER, "data_config.json"), 'w') as filepath:
        json.dump(generator.get_params(), filepath, indent=4)
