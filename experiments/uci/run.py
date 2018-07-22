# %%
import os
from project import EXPERIMENTS_PATH
from experiments.utils import write_config
from experiments.ranking import get_rankings
from experiments.uci import CONFIG, DATASET_CONFIG, ALGORITHMS

ID = DATASET_CONFIG["name"]
NAME = "uci"
FOLDER = os.path.join(EXPERIMENTS_PATH, NAME, "EXP_" + ID + "5")

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

write_results(FOLDER, relevances, durations, rankings)
relevances, durations, rankings = read_results(FOLDER)

# %%
# CALC ADDITIONAL STATISTICS (MEAN_DURCATIONS; CG; NDCG; SSE; MSE)
from experiments.utils import get_mean_durations
from experiments.metrics import compute_statistics, calc_avg_statistics

mean_durations = get_mean_durations(durations)
cgs, ndcgs, sses, mses = compute_statistics(rankings, relevances)
avg_ndcgs, gold_ranking = calc_avg_statistics(rankings)
print("\nOptimal ranking", list(gold_ranking.keys()))
avg_ndcgs

# %%
name = "avg_ndcg"
NEWFOLDER = os.path.join(FOLDER, name)
os.makedirs(NEWFOLDER, exist_ok=True)
ax = avg_ndcgs.plot(kind="line", title=name)
ax.set(xlabel="Missing Rate", ylabel="{:s} (AVG)".format(name))
fig = ax.get_figure()
fig.savefig(os.path.join(NEWFOLDER, "{:s}_means.png".format(name)))

# %%
# PLOT RESULTS
from experiments.plots import plot_mean_durations, plot_cgs, plot_scores

plot_mean_durations(FOLDER, mean_durations)
plot_scores(FOLDER, ndcgs, "NDCG")
plot_scores(FOLDER, sses, "SSE")
plot_scores(FOLDER, mses, "MSE")
plot_cgs(FOLDER, cgs)
