# %%
import os
import numpy as np
import pandas as pd
from time import clock
from copy import deepcopy
from collections import defaultdict

from experiments.utils import write_config
from experiments.NDCG_over_MissingRate import CONFIG, DATASET_CONFIG, ALGORITHMS

from project import EXPERIMENTS_PATH
from project.utils import introduce_missing_values, scale_data
from project.utils.data import DataGenerator
from project.utils.metrics import calculate_ndcg
from project.utils.imputer import Imputer

EXPERIMENT_ID = "0"
EXPERIMENT_NAME = "NDCG_over_MissingRate"
FOLDER = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME,
                      "EXP_" + EXPERIMENT_ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")
else:
    os.makedirs(FOLDER)

write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

relevances = []
results, durations = {}, {}

### CREATE DATASET WHICH IS USED FOR EVALUATION ###
for i in range(CONFIG["n_runs"]):
    generator = DataGenerator(**DATASET_CONFIG)
    data_original, relevance_vector = generator.create_dataset()
    data_original = scale_data(data_original)

    sorted_relevances = relevance_vector.sort_values(ascending=False)
    relevances.append(sorted_relevances)

    ### GATHER RESULTS FOR SPECIFIC MISSING RATE ###
    for missing_rate in CONFIG["missing_rates"]:
        results_run = defaultdict(list)
        durations_run = defaultdict(list)

        ### ADD MISSING VALUES TO DATASET (MULTIPLE TIMES) ###
        for j in range(CONFIG["n_insertions"]):
            data_orig = deepcopy(data_original)
            data_orig = introduce_missing_values(
                data_orig, missing_rate=missing_rate)

            for key, algorithm in ALGORITHMS.items():
                ### GET RANKING USING SELECTOR ###
                data = deepcopy(data_orig)
                start = clock()
                if algorithm["should_impute"]:
                    imputer = Imputer(data.f_types, algorithm["strategy"])
                    data = imputer.complete(data)

                selector = algorithm["class"](data.f_types, data.l_type,
                                              data.shape,
                                              **algorithm["config"])
                selector.fit(data.X, data.y)
                ranking = selector.get_ranking()
                duration = clock() - start

                ### EVALUATE RANKINGS AND STORE RESULTS ###
                ndcg = calculate_ndcg(relevance_vector, ranking)
                results_run[key].append(ndcg)
                durations_run[key].append(duration)

        # Update combined results
        if i == 0:
            results[missing_rate] = defaultdict(list)
            durations[missing_rate] = defaultdict(list)

        for key, scores in results_run.items():
            results[missing_rate][key].append(np.mean(scores))
        for key, times in durations_run.items():
            durations[missing_rate][key].append(np.mean(times))

        print("Finished missing rate {:.1f}".format(missing_rate), flush=True)
    print("Finished run {:d}".format(i + 1), flush=True)

# MEAN NDCG
mean_scores = {}
for key in ALGORITHMS.keys():
    mean_scores[key] = {}
    for missing_rate in CONFIG["missing_rates"]:
        mean_scores[key][missing_rate] = np.mean(results[missing_rate][key])
mean_scores = pd.DataFrame(mean_scores)

ax = mean_scores.plot(kind="line", title="NDCG over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="NDCG (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "ndcg_means.png"))

# STD NDCG
score_deviations = {}
for key in ALGORITHMS.keys():
    score_deviations[key] = {}
    for missing_rate in CONFIG["missing_rates"]:
        score_deviations[key][missing_rate] = np.std(
            results[missing_rate][key])
score_deviations = pd.DataFrame(score_deviations)

ax = score_deviations.plot(kind="line", title="NDCG over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="NDCG (Std)")
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "ndcg_deviations.png"))

# MEAN DURATIONS
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

############ Save stats ############
relevances = pd.DataFrame(relevances)
relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
mean_scores.to_csv(os.path.join(FOLDER, "mean_scores.csv"))
score_deviations.to_csv(os.path.join(FOLDER, "score_deviations.csv"))
mean_durations.to_csv(os.path.join(FOLDER, "runtimes.csv"))
