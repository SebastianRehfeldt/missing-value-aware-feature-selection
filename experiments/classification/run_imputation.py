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
from experiments.classification.utils import get_pipelines

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "ionosphere"
FOLDER = os.path.join(EXPERIMENTS_PATH, "classification", "imputation", name)
os.makedirs(FOLDER)

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

missing_rates = [0.2 * i for i in range(5)]
k_s = [2, 4, 6]  # [1, 2, 4, 6, 8]
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

                # STORE RESULTS
                print(names[i], "\n", res)
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

# %%
from glob import glob

paths = glob(FOLDER + "/*.csv")
results, missing_rates = [], []

for path in paths:
    results.append(pd.DataFrame.from_csv(path))
    missing_rates.append(path.split("_")[-1].split(".csv")[0])

avg_knn = pd.DataFrame()
for i, res in enumerate(results):
    avg_knn[missing_rates[i]] = res["AVG_knn"]

#ax = avg_knn.iloc[5:].T.plot(kind="line", title="F1 over missing rates")
ax = avg_knn.T.plot(kind="line", title="F1 over missing rates")
ax.set(xlabel="Missing Rate", ylabel="F1 (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "knn_means.png"))
avg_knn = pd.DataFrame()

time_knn = pd.DataFrame()
for i, res in enumerate(results):
    time_knn[missing_rates[i]] = res["TIME_knn"]

ax = time_knn.T.plot(kind="line", title="Fitting time over missing rates")
ax.set(xlabel="Missing Rate", ylabel="Time (s)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "knn_time.png"))
time_knn = pd.DataFrame()

avg_gnb = pd.DataFrame()
for i, res in enumerate(results):
    avg_gnb[missing_rates[i]] = res["AVG_gnb"]

#ax = avg_gnb.iloc[5:].T.plot(kind="line", title="F1 over missing rates")
ax = avg_gnb.T.plot(kind="line", title="F1 over missing rates")
ax.set(xlabel="Missing Rate", ylabel="F1 (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "gnb_means.png"))
