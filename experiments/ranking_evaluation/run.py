# %%
import os
from time import time
from project import EXPERIMENTS_PATH
from experiments.utils import write_config
from experiments.ranking import get_rankings, calc_mean_ranking
from experiments.ranking_evaluation import CONFIG, COMPETITORS, RAR
from experiments.ranking_evaluation import SYNTHETIC_CONFIG, UCI_CONFIG

t = time()

BASE_PATH = os.path.join(EXPERIMENTS_PATH, "ranking_evaluation")

# SELECT RAR VERSIONS OR COMPETITORS
ALGORITHMS = COMPETITORS

is_real_data = CONFIG["is_real_data"]
if is_real_data:
    DATASET_CONFIG = UCI_CONFIG
    ID = DATASET_CONFIG["name"] + "_comp"
    FOLDER = os.path.join(BASE_PATH, "uci", "EXP_" + ID)
else:
    ID = "single_comp"
    FOLDER = os.path.join(BASE_PATH, "synthetic", "EXP_" + ID)
    DATASET_CONFIG = SYNTHETIC_CONFIG

if os.path.isdir(FOLDER):
    print(FOLDER)
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

# CALC ADDITIONAL STATISTICS (MEAN_DURCATIONS; CG; NDCG; SSE; MSE)
from experiments.utils import get_mean_durations
from experiments.metrics import compute_statistics, calc_aucs

mean_durations = get_mean_durations(durations)
if is_real_data:
    statistics = compute_statistics(rankings, mean_scores, mean_scores)
else:
    statistics = compute_statistics(rankings, relevances, mean_scores)
cgs, cgs_at, ndcgs, cgs_pos, ndcgs_pos, sses, mses = statistics
aucs = calc_aucs(cgs_at[0])

# PLOT RESULTS
from experiments.plots import plot_mean_durations, plot_cgs, plot_scores
from experiments.plots import plot_aucs

plot_mean_durations(FOLDER, mean_durations)
plot_aucs(FOLDER, aucs)
plot_scores(FOLDER, cgs_at, "CG_AT_N")
plot_scores(FOLDER, ndcgs, "NDCG")
plot_scores(FOLDER, ndcgs_pos, "NDCG_POS")
plot_scores(FOLDER, sses, "SSE")
plot_scores(FOLDER, mses, "MSE")
plot_cgs(FOLDER, cgs, "CG")
plot_cgs(FOLDER, cgs_pos, "CG_POS")
print(time() - t)
