# %%
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from time import clock

from experiments.utils import write_config
from experiments.synthethic import CONFIG, DATASET_CONFIG, ALGORITHMS

from project import EXPERIMENTS_PATH
from project.utils import introduce_missing_values, scale_data
from project.utils.data import DataGenerator
from project.utils.metrics import calculate_cg, calculate_ndcg
from project.utils.imputer import Imputer
from project.utils.deleter import Deleter

ID = "0"
NAME = "synthethic"
FOLDER = os.path.join(EXPERIMENTS_PATH, NAME, "EXP_" + ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")
else:
    os.makedirs(FOLDER)

write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

durations, rankings = {}, {}

### CREATE DATASET WHICH IS USED FOR EVALUATION ###
for i in range(CONFIG["n_runs"]):
    generator = DataGenerator(**DATASET_CONFIG)
    generator.set_seed(CONFIG["seeds"][i])
    data_original, relevance_vector = generator.create_dataset()
    data_original = scale_data(data_original)

    if i == 0:
        relevances = pd.DataFrame(relevance_vector)
    else:
        relevances[i] = relevance_vector

    ### GATHER RESULTS FOR SPECIFIC MISSING RATE ###
    for missing_rate in CONFIG["missing_rates"]:
        durations_run = defaultdict(list)
        rankings_run = defaultdict(list)

        ### ADD MISSING VALUES TO DATASET (MULTIPLE TIMES) ###
        for j in range(CONFIG["n_insertions"]):
            data_orig = deepcopy(data_original)
            data_orig = introduce_missing_values(
                data_orig,
                missing_rate,
                seed=CONFIG["seeds"][j],
            )

            for key, algorithm in ALGORITHMS.items():
                ### GET RANKING USING SELECTOR ###
                data = deepcopy(data_orig)
                start = clock()
                if algorithm.get("should_impute", False):
                    imputer = Imputer(data.f_types, algorithm["strategy"])
                    data = imputer.complete(data)

                if algorithm.get("should_delete", False):
                    deleter = Deleter()
                    data = deleter.remove(data)

                selector = algorithm["class"](
                    data.f_types,
                    data.l_type,
                    data.shape,
                    **algorithm["config"],
                )
                selector.fit(data.X, data.y)
                ranking = selector.get_ranking()
                duration = clock() - start

                rankings_run[key].append(dict(ranking))
                durations_run[key].append(duration)

        # Update combined results
        if i == 0:
            durations[missing_rate] = defaultdict(list)
            rankings[missing_rate] = defaultdict(list)

        for key in ALGORITHMS.keys():
            durations[missing_rate][key].append(durations_run[key])
            rankings[missing_rate][key].append(rankings_run[key])

        print("Finished missing rate {:.1f}".format(missing_rate), flush=True)
    print("Finished run {:d}".format(i + 1), flush=True)

# %%
rankings

# %%
# STORE RAW RESULTS
from experiments.utils import write_json

relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
path = os.path.join(FOLDER, "runtimes.json")
write_json(durations, path)
path = os.path.join(FOLDER, "rankings.json")
write_json(rankings, path)

# %%
# SAVE PLOT FOR MEAN DURATIONS
from experiments.utils import plot_mean_durations

path = os.path.join(FOLDER, "runtimes.png")
plot_mean_durations(durations, path)

# %%

# %%
############ RESULTS FOR CG ############
CG_FOLDER = os.path.join(FOLDER, "CG")
os.makedirs(CG_FOLDER)
for mr in CONFIG["missing_rates"]:

    # MEAN CG
    mean_results = {
        key: np.mean(results_cg[mr][key], axis=0)
        for key in ALGORITHMS.keys()
    }
    cg_means = pd.DataFrame(mean_results)

    ax = cg_means.plot(kind="line", title="Cumulative Gain over features")
    ax.set(xlabel="# Features", ylabel="Cumulative Gain (Mean)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(CG_FOLDER, "cg_means{:.1f}.png").format(mr))

    # STD CG
    std_results = {
        key: np.std(results_cg[mr][key], axis=0)
        for key in ALGORITHMS.keys()
    }
    cg_stds = pd.DataFrame(std_results)

    ax = cg_stds.plot(kind="line", title="Cumulative Gain over features")
    ax.set(xlabel="# Features", ylabel="Cumulative Gain (Std)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(CG_FOLDER, "cg_stds{:.1f}.png").format(mr))

    # STATS
    cg_means.to_csv(os.path.join(CG_FOLDER, "cg_means{:.1f}.csv").format(mr))
    cg_stds.to_csv(os.path.join(CG_FOLDER, "cg_stds{:.1f}.csv").format(mr))

############ RESULTS FOR NDCG ############
NDCG_FOLDER = os.path.join(FOLDER, "NDCG")
os.makedirs(NDCG_FOLDER)
# MEAN NDCG
mean_scores = {}
for key in ALGORITHMS.keys():
    mean_scores[key] = {}
    for mr in CONFIG["missing_rates"]:
        mean_scores[key][mr] = np.mean(results_ndcg[mr][key])
mean_scores = pd.DataFrame(mean_scores)

ax = mean_scores.plot(kind="line", title="NDCG over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="NDCG (Mean)")
fig = ax.get_figure()
fig.savefig(os.path.join(NDCG_FOLDER, "ndcg_means.png"))

# STD NDCG
score_deviations = {}
for key in ALGORITHMS.keys():
    score_deviations[key] = {}
    for mr in CONFIG["missing_rates"]:
        score_deviations[key][mr] = np.std(results_ndcg[mr][key])
score_deviations = pd.DataFrame(score_deviations)

ax = score_deviations.plot(kind="line", title="NDCG over Missing Rate")
ax.set(xlabel="Missing Rate", ylabel="NDCG (Std)")
fig = ax.get_figure()
fig.savefig(os.path.join(NDCG_FOLDER, "ndcg_deviations.png"))

mean_scores.to_csv(os.path.join(NDCG_FOLDER, "mean_scores.csv"))
score_deviations.to_csv(os.path.join(NDCG_FOLDER, "score_deviations.csv"))
