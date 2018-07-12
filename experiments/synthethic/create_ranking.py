# %%
import os
import pandas as pd
from time import clock
from copy import deepcopy
from collections import defaultdict

from experiments.utils import write_config
from experiments.synthethic import CONFIG, DATASET_CONFIG, ALGORITHMS

from project import EXPERIMENTS_PATH
from project.utils.data import DataGenerator
from project.utils import introduce_missing_values, scale_data
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
# STORE RAW RESULTS
from experiments.utils import write_json

relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
path = os.path.join(FOLDER, "runtimes.json")
write_json(durations, path)
path = os.path.join(FOLDER, "rankings.json")
write_json(rankings, path)

# %%
# READ RAW RESULTS
from experiments.utils import read_json

path = os.path.join(FOLDER, "rankings.json")
rankings = read_json(path)
path = os.path.join(FOLDER, "relevances.csv")
relevances = pd.DataFrame.from_csv(path)
path = os.path.join(FOLDER, "runtimes.json")
durations = read_json(path)

# %%
# CALC ADDITIONAL STATISTICS (MEAN_DURCATIONS; CG; NDCG; SSE)
from experiments.utils import get_mean_durations
from experiments.utils import compute_gain_statistics

mean_durations = get_mean_durations(durations)
cgs, ndcgs = compute_gain_statistics(rankings, relevances)

# %%
############ PLOT RESULTS FOR NDCG ############
from experiments.plots import plot_mean_durations
from experiments.plots import plot_ndcgs, plot_cgs

plot_mean_durations(mean_durations, FOLDER)
plot_ndcgs(ndcgs, FOLDER)
plot_cgs(cgs, FOLDER)

# %%
############ RESULTS FOR SSE ############