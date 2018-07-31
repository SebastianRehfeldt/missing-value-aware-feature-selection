# %%
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

from sklearn.cross_validation import cross_val_score, StratifiedKFold

from project import EXPERIMENTS_PATH
from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data
from experiments.pipelines.utils import get_pipelines

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "ionosphere"
FOLDER = os.path.join(EXPERIMENTS_PATH, "pipelines", name)
os.makedirs(FOLDER, exist_ok=True)

data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
data = scale_data(data)
data.shuffle_rows(seed=42)

names = [
    "rar",
    "rar + impute",
]

missing_rates = [0.2]
k_s = [2]
classifiers = ["knn"]
for mr in missing_rates:
    data_copy = deepcopy(data)
    data_copy = introduce_missing_values(data_copy, missing_rate=mr, seed=42)
    scores = []

    for clf in classifiers:
        scores_clf = pd.DataFrame()

        for k in k_s:
            cv = StratifiedKFold(data_copy.y, n_folds=3)

            pipelines = get_pipelines(data_copy, k, names, clf)
            for i, pipe in enumerate(pipelines):
                start = time()
                res = cross_val_score(
                    pipe,
                    data_copy.X,
                    data_copy.y,
                    cv=cv,
                    scoring="f1_micro",
                )
                mean, std, t = np.mean(res), np.std(res), time() - start
                col = names[i]
                if not col == "complete":
                    col += "_" + str(k)
                scores_clf[col] = pd.Series({
                    "AVG_" + clf: mean,
                    "STD_" + clf: std,
                    "TIME_" + clf: t
                })
        scores.append(scores_clf)
    scores = pd.concat(scores).T
    scores.to_csv(os.path.join(FOLDER, "results_{:.2f}.csv".format(mr)))
