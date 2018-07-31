# %%
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy

from sklearn.metrics import f1_score

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
    "rar ++ impute mice",
    "rar + impute mice",
    "mice impute + rar",
    "mice impute ++ rar",
]

missing_rates = [0.2 * i for i in range(3)]
k_s = [2, 6]  # [1, 2, 4, 6, 8]
classifiers = ["knn", "gnb"]
for mr in missing_rates:
    data_copy = deepcopy(data)
    data_copy = introduce_missing_values(data_copy, missing_rate=mr, seed=42)
    scores = []

    for clf in classifiers:
        scores_clf = pd.DataFrame()

        for k in k_s:
            pipelines = get_pipelines(data_copy, k, names, clf)

            for i, pipe in enumerate(pipelines):
                # GET RESULTS
                start = time()
                if clf in ["gnb"] and mr > 0 and "impute" not in names[i]:
                    mean, std, t = 0, 0, 0
                else:
                    res = []
                    splits = data_copy.split()
                    for train, test in splits:
                        reducer = pipe.named_steps.get("reduce")
                        if reducer is not None:
                            reducer.set_params(is_fitted=False)

                        pipe.fit(train.X, train.y)

                        if "++" in names[i]:
                            temp_step = deepcopy(pipe.steps[0])
                            pipe.steps[0] = deepcopy(pipe.steps[1])
                            pipe.steps[1] = temp_step

                        y_pred = pipe.predict(test.X)
                        f1 = f1_score(test.y, y_pred, average="micro")
                        res.append(f1)

                        if "++" in names[i]:
                            temp_step = deepcopy(pipe.steps[0])
                            pipe.steps[0] = deepcopy(pipe.steps[1])
                            pipe.steps[1] = temp_step

                    mean, std, t = np.mean(res), np.std(res), time() - start

                print(names[i], "\n", res)
                # STORE RESULTS
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
