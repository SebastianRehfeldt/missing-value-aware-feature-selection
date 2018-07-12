# %%
import os
from project import EXPERIMENTS_PATH
from experiments.utils import write_config
from experiments.ranking import get_rankings
from experiments.uci import CONFIG, DATASET_CONFIG, ALGORITHMS

ID = "0"
NAME = "uci"
FOLDER = os.path.join(EXPERIMENTS_PATH, NAME, "EXP_" + ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")

os.makedirs(FOLDER)
write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

# GET RANKINGS
res = get_rankings(CONFIG, DATASET_CONFIG, ALGORITHMS)
rankings, durations, relevances = res

# STORE AND READ RAW RESULTS
from experiments.utils import write_results, read_results

write_results(FOLDER, relevances, durations, rankings)
relevances, durations, rankings = read_results(FOLDER)

# %%
"""
# CALC ADDITIONAL STATISTICS (MEAN_DURCATIONS; CG; NDCG; SSE; MSE)
from experiments.utils import get_mean_durations
from experiments.metrics import compute_statistics

mean_durations = get_mean_durations(durations)
cgs, ndcgs, sses, mses = compute_statistics(rankings, relevances)

# PLOT RESULTS
from experiments.plots import plot_mean_durations, plot_cgs, plot_scores

plot_mean_durations(FOLDER, mean_durations)
plot_scores(FOLDER, ndcgs, "NDCG")
plot_scores(FOLDER, sses, "SSE")
plot_scores(FOLDER, mses, "MSE")
plot_cgs(FOLDER, cgs)
"""
