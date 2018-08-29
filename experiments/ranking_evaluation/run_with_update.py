# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project import EXPERIMENTS_PATH
from experiments.utils import write_config
from experiments.ranking import get_rankings, calc_mean_ranking
from experiments.ranking_evaluation import CONFIG, RAR, COMPETITORS
from experiments.ranking_evaluation import SYNTHETIC_CONFIG as DATASET_CONFIG

ALGORITHMS = COMPETITORS

ID = "half_discrete_3-1_comp"
FOLDER = os.path.join(EXPERIMENTS_PATH, "ranking_evaluation", "updates",
                      "EXP_" + ID)
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
mean_scores = calc_mean_ranking(rankings)

# READ CONFIG AND PREPARE INDEX
n_mrs = len(durations.keys())
algorithms = list(durations['0.0'].keys())
n_algorithms = len(algorithms)
n_runs = len(durations['0.0'][algorithms[0]])
update = CONFIG["update_attribute"]
index = CONFIG["updates"]
index = [d[update] for d in index]

# COMPUTE MEAN DURATIONS
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

# PLOT AND STORE MEAN DURATIONS
mean_durations.to_csv(os.path.join(FOLDER, "mean_runtime.csv"))
ax = mean_durations.plot(kind="bar", title="Runtime over {:s}".format(update))
ax.set(xlabel=update, ylabel="Runtime (s)")
ax.xaxis.set_tick_params(rotation=0)
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "mean_runtimes.png").format(mr))
plt.close(fig)

# COMPUTE AND PLOT STATISTICS FOR SINGLE RUNS
from experiments.metrics import compute_statistics, calc_aucs
from experiments.plots import plot_cgs, plot_scores, plot_aucs

stats = [None] * n_runs
for run in range(n_runs):
    stats[run] = compute_statistics(rankings, relevances, mean_scores, run)
    cgs, cgs_at, ndcgs, cgs_pos, ndcgs_pos, sses, mses = stats[run]
    aucs = calc_aucs(cgs_at[0])

    NEW_FOLDER = os.path.join(FOLDER, update + "_" + str(index[run]))

    plot_aucs(NEW_FOLDER, aucs)
    plot_scores(NEW_FOLDER, cgs_at, "CG_AT_N")
    plot_scores(NEW_FOLDER, ndcgs, "NDCG")
    plot_scores(NEW_FOLDER, ndcgs_pos, "NDCG_POS")
    plot_scores(NEW_FOLDER, sses, "SSE")
    plot_scores(NEW_FOLDER, mses, "MSE")
    plot_cgs(NEW_FOLDER, cgs, "CG")
    plot_cgs(NEW_FOLDER, cgs_pos, "CG_POS")

# COMPUTE NDCG OVER DATASET UPDATES
ndcg_mean, ndcg_deviation = {}, {}
cg_mean, cg_deviation = {}, {}
for run in range(n_runs):
    cgs, cgs_at, ndcgs, cgs_pos, ndcgs_pos, sses, mses = stats[run]
    ndcg_mean[run] = ndcgs[0].mean(0)
    ndcg_deviation[run] = ndcgs[0].std(0)
    cg_mean[run] = cgs_at[0].mean(0)
    cg_deviation[run] = cgs_at[0].std(0)

# PLOT AND STORE NDCG OVER DATASETS AND MISSING RATES
ndcg_mean = pd.DataFrame(ndcg_mean).T
ndcg_mean.index = index
ndcg_deviation = pd.DataFrame(ndcg_deviation).T
ndcg_deviation.index = index

kind = "bar" if n_runs <= 2 else "line"

ndcg_mean.to_csv(os.path.join(FOLDER, "mean_ndcgs.csv"))
ax = ndcg_mean.plot(
    kind=kind,
    title="NDCG (mean) over {:s} and missing rates".format(update),
)
ax.set(xlabel=update, ylabel="NDCG (mean)")
ax.xaxis.set_tick_params(rotation=0)
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "ndcg_mean.png"))
plt.close(fig)

ndcg_deviation.to_csv(os.path.join(FOLDER, "mean_deviation.csv"))
ax = ndcg_deviation.plot(
    kind=kind,
    title="NDCG (std) over {:s} and missing rates".format(update),
)
ax.set(xlabel=update, ylabel="NDCG (std)")
ax.xaxis.set_tick_params(rotation=0)
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "ndcg_deviation.png"))
plt.close(fig)

# PLOT AND STORE CG AT OVER DATASETS AND MISSING RATES
cg_mean = pd.DataFrame(cg_mean).T
cg_mean.index = index
cg_deviation = pd.DataFrame(cg_deviation).T
cg_deviation.index = index

kind = "bar" if n_runs <= 2 else "line"

cg_mean.to_csv(os.path.join(FOLDER, "mean_cgs.csv"))
ax = cg_mean.plot(
    kind=kind,
    title="CG (mean) over {:s} and missing rates".format(update),
)
ax.set(xlabel=update, ylabel="CG (mean)")
ax.xaxis.set_tick_params(rotation=0)
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "cg_mean.png"))
plt.close(fig)

cg_deviation.to_csv(os.path.join(FOLDER, "deviation_cgs.csv"))
ax = cg_deviation.plot(
    kind=kind,
    title="CG (std) over {:s} and missing rates".format(update),
)
ax.set(xlabel=update, ylabel="CG (std)")
ax.xaxis.set_tick_params(rotation=0)
fig = ax.get_figure()
fig.savefig(os.path.join(FOLDER, "cg_deviation.png"))
plt.close(fig)
