# %%
import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from time import clock

from experiments.utils import write_config
from experiments.fscore import CONFIG, DATASET_CONFIG, ALGORITHMS

from project import EXPERIMENTS_PATH
from project.utils import introduce_missing_values, scale_data, DataLoader
from project.utils.imputer import Imputer
from project.utils.deleter import Deleter

from project.classifier import KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
scorer = make_scorer(f1_score, average="micro")

EXPERIMENT_ID = "8"
EXPERIMENT_NAME = "fscore"
FOLDER = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME,
                      "EXP_" + EXPERIMENT_ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")
else:
    os.makedirs(FOLDER)

write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

durations = {}
rankings = {}
fscores = {}
fscores_bayes = {}
fscores_knn = {}

k = 4

### CREATE DATASET WHICH IS USED FOR EVALUATION ###
datasets = ["ionosphere"]
for dataset in datasets:
    data_loader = DataLoader()
    data_original = data_loader.load_data(dataset, "arff")
    data_original = scale_data(data_original)

    ### GATHER RESULTS FOR SPECIFIC MISSING RATE ###
    for missing_rate in CONFIG["missing_rates"]:
        durations_run = defaultdict(list)
        rankings_run = defaultdict(list)
        fscores[missing_rate] = defaultdict(list)
        fscores_bayes[missing_rate] = defaultdict(list)
        fscores_knn[missing_rate] = defaultdict(list)

        ### ADD MISSING VALUES TO DATASET (MULTIPLE TIMES) ###
        for j in range(CONFIG["n_insertions"]):
            data_orig = deepcopy(data_original)
            data_orig = introduce_missing_values(
                data_orig, missing_rate=missing_rate)

            cv = StratifiedKFold(data_orig.y, n_folds=3, shuffle=True)

            for key, algorithm in ALGORITHMS.items():
                ### GET RANKING USING SELECTOR ###
                data = deepcopy(data_orig)
                start = clock()
                if algorithm["should_impute"]:
                    imputer = Imputer(data.f_types, algorithm["strategy"])
                    data = imputer.complete(data)

                if algorithm.get("should_delete", False):
                    deleter = Deleter()
                    data = deleter.remove(data)

                selector = algorithm["class"](data.f_types, data.l_type,
                                              data.shape,
                                              **algorithm["config"])
                selector.fit(data.X, data.y)
                importances = selector.feature_importances
                duration = clock() - start

                ### EVALUATE RANKINGS AND STORE RESULTS ###
                durations_run[key].append(duration)
                rankings_run[key].append(importances)

                X_new = selector.transform(data_orig.X, k)
                knn = KNN(data.f_types, data.l_type)
                gnb = GaussianNB()

                scores = cross_val_score(
                    knn, X_new, data.y, cv=cv, scoring=scorer)
                fscores[missing_rate][key].append(np.mean(scores))

                imputer = Imputer(data.f_types, "mice")
                data = imputer.complete(data)
                X_imputed = selector.transform(data.X, k)

                scores_knn = cross_val_score(
                    knn, X_imputed, data.y, cv=cv, scoring=scorer)
                fscores_knn[missing_rate][key].append(np.mean(scores_knn))

                scores_bayes = cross_val_score(
                    gnb, X_imputed, data.y, cv=cv, scoring=scorer)
                fscores_bayes[missing_rate][key].append(np.mean(scores_bayes))

        # Update combined results
        rankings[missing_rate] = rankings_run
        durations[missing_rate] = durations_run

        print("Finished missing rate {:.1f}".format(missing_rate), flush=True)
    print("Finished dataset {:s}".format(dataset), flush=True)

############ F_scores ############
mean_scores = {}
for missing_rate, results in fscores.items():
    mean_scores[missing_rate] = defaultdict(list)
    for algorithm, scores in results.items():
        mean_scores[missing_rate][algorithm] = np.mean(scores)
df = pd.DataFrame(mean_scores).T
df.to_csv(os.path.join(FOLDER, "fscores.csv"))

ax = df.plot(kind="line", title="F-score over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="F-score (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "fscores.png"))

############ F_scores ############
mean_scores = {}
for missing_rate, results in fscores_bayes.items():
    mean_scores[missing_rate] = defaultdict(list)
    for algorithm, scores in results.items():
        mean_scores[missing_rate][algorithm] = np.mean(scores)
df = pd.DataFrame(mean_scores).T
df.to_csv(os.path.join(FOLDER, "fscores_bayes.csv"))

ax = df.plot(kind="line", title="F-score over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="F-score (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "fscores_bayes.png"))

############ F_scores ############
mean_scores = {}
for missing_rate, results in fscores_knn.items():
    mean_scores[missing_rate] = defaultdict(list)
    for algorithm, scores in results.items():
        mean_scores[missing_rate][algorithm] = np.mean(scores)
df = pd.DataFrame(mean_scores).T
df.to_csv(os.path.join(FOLDER, "fscores_knn.csv"))

ax = df.plot(kind="line", title="F-score over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="F-score (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "fscores_knn.png"))

############ RUNTIME ############
mean_durations = {}
for key in ALGORITHMS.keys():
    mean_durations[key] = {}
    for missing_rate in CONFIG["missing_rates"]:
        mean_durations[key][missing_rate] = np.mean(
            durations[missing_rate][key])
mean_durations = pd.DataFrame(mean_durations)

ax = mean_durations.plot(kind="bar", title="Mean fitting time", rot=0)
ax.set(xlabel="Missing Rate", ylabel="Time in seconds")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "runtimes.png"))

mean_durations.to_csv(os.path.join(FOLDER, "runtimes.csv"))
