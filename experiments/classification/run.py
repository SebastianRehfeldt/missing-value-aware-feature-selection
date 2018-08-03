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
from project.utils import DataLoader, Data
from project.utils import introduce_missing_values, scale_data
from experiments.classification.utils import get_selectors, get_classifiers
from experiments.plots import plot_mean_durations

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "ionosphere"
FOLDER = os.path.join(EXPERIMENTS_PATH, "classification", "incomplete", name)
CSV_FOLDER = os.path.join(FOLDER, "csv")
os.makedirs(FOLDER)
os.makedirs(CSV_FOLDER)

data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
data = scale_data(data)
data.shuffle_rows(seed=42)

names = [
    "rar", "rknn", "sfs", "mi", "mrmr", "cfs", "relief_o", "fcbf_o", "rf",
    "xgb"
]

seeds = [42, 0, 13]
n_runs = 3 if len(seeds) >= 3 else len(seeds)
n_insertions = 3 if len(seeds) >= 3 else len(seeds)
classifiers = ["knn", "tree", "gnb", "svm"]
k_s = [i + 1 for i in range(15)]
missing_rates = [0.1 * i for i in range(10)]

times = {mr: defaultdict(list) for mr in missing_rates}
complete_scores = deepcopy(times)
for mr in missing_rates:

    scores_clf = {k: defaultdict(list) for k in k_s}
    scores = {algo: deepcopy(scores_clf) for algo in classifiers}

    for j in range(n_insertions):
        d = deepcopy(data)
        d = introduce_missing_values(d, missing_rate=mr, seed=seeds[j])

        splits = d.split(n_repeats=n_runs)
        for i_split, (train, test) in enumerate(splits):
            # EVALUATE COMPLETE SET
            clfs = get_classifiers(train, classifiers)
            for i_c, clf in enumerate(clfs):
                clf.fit(train.X, train.y)
                y_pred = clf.predict(test.X)
                f1 = f1_score(test.y, y_pred, average="micro")
                complete_scores[mr][classifiers[i_c]].append(f1)

            # EVALUATE SELECTORS
            selectors = get_selectors(train, names, max(k_s))
            for i_s, selector in enumerate(selectors):
                start = time()

                train_data = deepcopy(train)
                np.random.seed(seeds[i_split % n_runs])
                selector.fit(train_data.X, train_data.y)

                t = time() - start
                times[mr][names[i_s]].append(t)

                for k in k_s:
                    X_train = selector.transform(train.X, k)
                    X_test = selector.transform(test.X, k)

                    f_types = train.f_types[X_train.columns]
                    transformed_data = Data(X_train, train.y, f_types,
                                            train.l_type, X_train.shape)

                    clfs = get_classifiers(transformed_data, classifiers)
                    for i_c, clf in enumerate(clfs):
                        clf.fit(X_train, train.y.reset_index(drop=True))
                        y_pred = clf.predict(X_test)
                        f1 = f1_score(test.y, y_pred, average="micro")
                        scores[classifiers[i_c]][k][names[i_s]].append(f1)

    for clf in classifiers:
        means = pd.DataFrame(scores[clf]).applymap(np.mean).T
        stds = pd.DataFrame(scores[clf]).applymap(np.std).T
        means.to_csv(
            os.path.join(CSV_FOLDER, "mean_{:s}_{:.2f}.csv".format(clf, mr)))
        stds.to_csv(
            os.path.join(CSV_FOLDER, "std_{:s}_{:.2f}.csv".format(clf, mr)))

# TIMES
mean_times = pd.DataFrame(times).applymap(np.mean).T
mean_times.to_csv(os.path.join(CSV_FOLDER, "mean_times.csv"))
std_times = pd.DataFrame(times).applymap(np.std).T
std_times.to_csv(os.path.join(CSV_FOLDER, "std_times.csv"))

# COMPLETE SCORES
mean_scores = pd.DataFrame(complete_scores).applymap(np.mean).T
mean_scores.to_csv(os.path.join(CSV_FOLDER, "mean_scores.csv"))
std_scores = pd.DataFrame(complete_scores).applymap(np.std).T
std_scores.to_csv(os.path.join(CSV_FOLDER, "std_scores.csv"))

# PLOT TIMES
times = pd.DataFrame.from_csv(os.path.join(CSV_FOLDER, "mean_times.csv"))
plot_mean_durations(FOLDER, times)

times = pd.DataFrame.from_csv(os.path.join(CSV_FOLDER, "std_times.csv"))
ax = times.plot(kind="line", title="Fitting time over missing rates")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "runtimes_deviations.png"))
times = pd.DataFrame()

# PLOT AVG AMONG BEST K=5 FEATURES
k = 5
file_prefixes = ["mean_", "std_"]

mean_scores = pd.DataFrame.from_csv(
    os.path.join(CSV_FOLDER, "mean_scores.csv"))
std_scores = pd.DataFrame.from_csv(os.path.join(CSV_FOLDER, "std_scores.csv"))

for clf in classifiers:
    for p in file_prefixes:
        search_string = "{:s}{:s}*.csv".format(p, clf)
        filepaths = glob(os.path.join(CSV_FOLDER, search_string))

        for i, f in enumerate(filepaths):
            df = pd.DataFrame.from_csv(f)
            if i == 0:
                scores = pd.DataFrame(
                    np.zeros((len(missing_rates), len(df.columns) + 1)),
                    index=missing_rates,
                    columns=["complete"] + list(df.columns))

            scores.iloc[i] = df.iloc[:k].mean()

        scores["complete"] = mean_scores[clf] if "me" in p else std_scores[clf]

        t = "Average f1 {:s} among top {:d} features ({:s})".format(p, k, clf)
        ax = scores.plot(kind="line", title=t)
        ax.set(xlabel="Missing Rate", ylabel="F1")
        fig = ax.get_figure()
        filename = "{:s}f1_{:s}_{:d}.png".format(p, clf, k)
        fig.savefig(os.path.join(FOLDER, filename))

# PLOT SINGLE FILES
mr_s = [0.00]
clfs = ["knn", "tree"]
kinds = ["mean", "std"]

for clf in clfs:
    for mr in mr_s:
        title = "F1 score over features (clf={:s}, mr={:.2f})".format(clf, mr)
        for kind in kinds:
            path = os.path.join(CSV_FOLDER, "{:s}_{:s}_{:.2f}.csv".format(
                kind, clf, mr))
            df = pd.DataFrame.from_csv(path)

            if kind == "mean":
                df["complete"] = mean_scores[clf][mr]
            else:
                df["complete"] = std_scores[clf][mr]

            ax = df.plot(title=title)
            ax.set(xlabel="# Features", ylabel="F1 ({:s})".format(kind))
            fig = ax.get_figure()
            path = "{:s}_{:s}_{:.2f}.png".format(clf, kind, mr)
            fig.savefig(os.path.join(FOLDER, path))
