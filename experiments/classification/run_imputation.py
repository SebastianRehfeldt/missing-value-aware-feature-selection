# %%
import os
import numpy as np
import pandas as pd
from time import time
from glob import glob
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import f1_score

from project import EXPERIMENTS_PATH
from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data
from experiments.classification.utils import get_pipelines, swap_pipeline_steps
from experiments.plots import plot_mean_durations

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "heart-c"
FOLDER = os.path.join(EXPERIMENTS_PATH, "classification", "imputation", name)
os.makedirs(FOLDER)

data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
data = scale_data(data)
data.shuffle_rows(seed=42)

names = [
    "rar",
    "rar + impute mice",
    "rar ++ impute mice",
    "mice impute + rar",
    "mice impute ++ rar",
]

k_s = [i + 1 for i in range(10)]
seeds = [42, 0, 13]
n_runs = 3 if len(seeds) >= 3 else len(seeds)
n_insertions = 3 if len(seeds) >= 3 else len(seeds)
missing_rates = [0.2 * i for i in range(5)]
classifiers = ["knn", "tree", "gnb", "svm"]

# MISSING RATES
for mr in missing_rates:
    scores = []

    # CLASSIFIERS
    for clf in classifiers:
        scores_clf = defaultdict(list)
        times_clf = defaultdict(list)

        # INSERTIONS
        for j in range(n_insertions):
            d = deepcopy(data)
            d = introduce_missing_values(d, missing_rate=mr, seed=seeds[j])

            # SPLITS AND RUNS
            splits = d.split(n_repeats=n_runs)
            for i_split, (train_data, test_data) in enumerate(splits):

                # K'S
                for k in k_s:

                    # PIPELINES
                    pipelines = get_pipelines(train_data, d, k, names, clf)
                    for i, pipe in enumerate(pipelines):
                        # GET RESULTS
                        robust = clf not in ["gnb"] or "impute" in names[i]
                        if not robust and mr > 0:
                            f1, t = 0, 0
                        else:
                            train = deepcopy(train_data)
                            test = deepcopy(test_data)
                            start = time()

                            seed = seeds[i_split % n_runs]
                            np.random.seed(seed)
                            pipe.fit(train.X, train.y.reset_index(drop=True))

                            if "++" in names[i]:
                                swap_pipeline_steps(pipe)

                            y_pred = pipe.predict(test.X)
                            f1 = f1_score(test.y, y_pred, average="micro")
                            t = time() - start

                        # STORE RESULTS
                        col = names[i]
                        if not col == "complete":
                            col += "_" + str(k)

                        scores_clf[col].append(f1)
                        times_clf[col].append(t)

        mean_scores = pd.DataFrame(scores_clf).mean()
        std_scores = pd.DataFrame(scores_clf).std()
        mean_times = pd.DataFrame(times_clf).mean()
        scores_clf = pd.DataFrame({
            "AVG_{:s}".format(clf): mean_scores,
            "STD_{:s}".format(clf): std_scores,
            "TIME_{:s}".format(clf): mean_times,
        }).T
        scores.append(scores_clf)
    scores = pd.concat(scores).T
    scores.to_csv(os.path.join(FOLDER, "results_{:.2f}.csv".format(mr)))

# READ RESULTS
paths = glob(FOLDER + "/*.csv")
results, missing_rates = [], []

for path in paths:
    results.append(pd.read_csv(path))
    missing_rates.append(path.split("_")[-1].split(".csv")[0])

# PLOT TIME
time_knn = pd.DataFrame()
for i, res in enumerate(results):
    time_knn[missing_rates[i]] = res["TIME_knn"]
plot_mean_durations(FOLDER, time_knn.T)

# PLOT CLASSIFICATION SCORES
for clf in classifiers:
    scores = pd.DataFrame()
    for i, res in enumerate(results):
        scores[missing_rates[i]] = res["AVG_{:s}".format(clf)]

    # ax = scores.iloc[5:].T.plot(kind="line", title="F1 over missing rates")
    ax = scores.T.plot(kind="line", title="F1 over missing rates")
    ax.set(xlabel="Missing Rate", ylabel="F1 (Mean)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(FOLDER, "{:s}_means.png".format(clf)))
