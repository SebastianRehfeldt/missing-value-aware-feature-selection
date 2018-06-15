# %%
import os
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from time import clock

from experiments.utils import write_config
from experiments.CG_over_Features import CONFIG, DATASET_CONFIG, ALGORITHMS

from project import EXPERIMENTS_PATH
from project.utils import introduce_missing_values, scale_data
from project.utils.data import DataGenerator
from project.utils.metrics import calculate_cg
from project.utils.imputer import Imputer

EXPERIMENT_ID = "1"
EXPERIMENT_NAME = "CG_over_Features"
FOLDER = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME,
                      "EXP_" + EXPERIMENT_ID)
if os.path.isdir(FOLDER):
    raise Exception("Set experiment id to run new experiment")
else:
    os.makedirs(FOLDER)

write_config(FOLDER, CONFIG, DATASET_CONFIG, ALGORITHMS)

for missing_rate in CONFIG["missing_rates"]:
    results = defaultdict(list)
    durations = defaultdict(list)
    relevances = []
    generator = DataGenerator(**DATASET_CONFIG)

    ############ GATHER RESULTS ############
    for i in range(CONFIG["n_runs"]):
        data_original, relevance_vector = generator.create_dataset()
        data_original = scale_data(data_original)

        sorted_relevances = relevance_vector.sort_values(ascending=False)
        relevances.append(sorted_relevances)

        for j in range(CONFIG["n_insertions"]):
            data_orig = deepcopy(data_original)
            data_orig = introduce_missing_values(
                data_orig, missing_rate=missing_rate)

            for key, algorithm in ALGORITHMS.items():
                data = deepcopy(data_orig)

                start = clock()
                if algorithm["should_impute"]:
                    imputer = Imputer(data.f_types, algorithm["strategy"])
                    data = imputer.complete(data)

                selector = algorithm["class"](data.f_types, data.l_type,
                                              data.shape,
                                              **algorithm["config"])
                selector.fit(data.X, data.y)
                durations[key].append(clock() - start)

                ranking = selector.get_ranking()
                CG = calculate_cg(relevance_vector, ranking)
                results[key].append(CG)

        print("Finished run {:d}".format(i + 1), flush=True)

    ############ PLOTS ############
    # MEAN CG
    mean_results = {
        key: np.mean(results[key], axis=0)
        for key in ALGORITHMS.keys()
    }
    cg_means = pd.DataFrame(mean_results)

    ax = cg_means.plot(kind="line", title="Cumulative Gain over features")
    ax.set(xlabel="# Features", ylabel="Cumulative Gain (Mean)")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(FOLDER, "cg_means{:.1f}.png").format(missing_rate))

    # STD CG
    std_results = {
        key: np.std(results[key], axis=0)
        for key in ALGORITHMS.keys()
    }
    cg_deviations = pd.DataFrame(std_results)
    ax = cg_deviations.plot(kind="line", title="Cumulative Gain over features")
    ax.set(xlabel="# Features", ylabel="Cumulative Gain (Std)")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(FOLDER, "cg_deviations{:.1f}.png").format(missing_rate))

    # MEAN TIME
    duration_means = {
        key: [np.mean(durations[key], axis=0)]
        for key in ALGORITHMS.keys()
    }
    duration_means = pd.DataFrame(duration_means, index=["Algorithms"])
    ax = duration_means.plot(kind="bar", title="Mean fitting time", rot=0)
    ax.set(ylabel="Time in seconds")
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(FOLDER, "runtimes{:.1f}.png").format(missing_rate))

    ############ Save stats ############
    relevances = pd.DataFrame(relevances)
    relevances.to_csv(
        os.path.join(FOLDER, "relevances{:.1f}.csv").format(missing_rate))
    cg_means.to_csv(
        os.path.join(FOLDER, "cg_means{:.1f}.csv").format(missing_rate))
    cg_deviations.to_csv(
        os.path.join(FOLDER, "cg_deviations{:.1f}.csv").format(missing_rate))
    duration_means.to_csv(
        os.path.join(FOLDER, "runtimes{:.1f}.csv").format(missing_rate))
