# %%
import os
import numpy as np
import pandas as pd
from project import EXPERIMENTS_PATH
from experiments.utils import write_config
from experiments.ranking import get_rankings, calc_mean_ranking
from experiments.synthethic import CONFIG, DATASET_CONFIG, ALGORITHMS

ID = "100"
NAME = "synthethic"
FOLDER = os.path.join(EXPERIMENTS_PATH, NAME, "EXP_" + ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")

os.makedirs(FOLDER)
write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

# GET RANKINGS
res = get_rankings(CONFIG, DATASET_CONFIG, ALGORITHMS)
rankings, durations, relevances = res

# %%
# STORE AND READ RAW RESULTS
from experiments.utils import write_results, read_results

# write_results(FOLDER, relevances, durations, rankings)
relevances, durations, rankings = read_results(FOLDER)
mean_scores = calc_mean_ranking(rankings)

# %%
n_mrs = len(durations.keys())
algorithms = list(durations['0.0'].keys())
n_algorithms = len(algorithms)
n_runs = len(durations['0.0'][algorithms[0]])
index = CONFIG["updates"]
index = [d["n_samples"] for d in index]
index = [500, 2000]

mean_durations = pd.DataFrame(
    np.zeros((n_runs, n_algorithms)),
    columns=algorithms,
    index=index,
)

for mr, results in durations.items():
    for algorithm, times in results.items():
        for run in range(len(times)):
            t = np.mean(times[run]) / n_mrs
            mean_durations[algorithm].iloc[run] += t

ax = mean_durations.plot(kind="bar", title="Runtime over samples")
ax.set(xlabel="# Samples", ylabel="Runtime (s)")
ax.xaxis.set_tick_params(rotation=0)

# %%
rankings
