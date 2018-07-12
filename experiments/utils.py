import os
import json
import numpy as np
import pandas as pd

from experiments.metrics import calculate_cg, calculate_ndcg


def write_config(folder, config, dataset_config, algorithms):
    filename = os.path.join(folder, "config.txt")
    with open(filename, 'w+') as file:
        file.write("CONFIG EXPERIMENT\n\n")
        file.write(json.dumps(config, indent=4))
        file.write("\n\nCONFIG DATASET\n\n")
        file.write(json.dumps(dataset_config, indent=4))
        file.write("\n\nCONFIG ALGORITHMS")

        for key, algorithm in algorithms.items():
            shape = (dataset_config["n_samples"], dataset_config["n_features"])
            data = [[], "", shape]
            selector = algorithm["class"](*data, **algorithm["config"])
            params = selector.get_params()
            del params["f_types"]
            del params["l_type"]
            del params["shape"]
            file.write("\n\nCONFIG - {:s}\n".format(key))
            file.write(json.dumps(params, indent=4))


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def get_mean_durations(durations):
    mean_durations = {}
    for missing_rate in durations.keys():
        mean_durations[missing_rate] = {}

        for key in durations[missing_rate].keys():
            mean_time = np.mean(durations[missing_rate][key], axis=1)
            mean_time = np.mean(mean_time)
            mean_durations[missing_rate][key] = mean_time

    return pd.DataFrame(mean_durations).T


def compute_gain_statistics(rankings, relevances):
    cg_means, cg_deviations = {}, {}
    ndcg_means, ndcg_deviations = {}, {}
    for missing_rate in rankings.keys():
        cg_means[missing_rate] = {}
        cg_deviations[missing_rate] = {}
        ndcg_means[missing_rate] = {}
        ndcg_deviations[missing_rate] = {}

        for key in rankings[missing_rate].keys():
            ranking = rankings[missing_rate][key]

            # The mean and std are calculated over all datasets and insertions
            # Run means a new dataset and i indicates multiple insertions
            cgs, ndcgs = [], []
            for run in range(len(ranking)):
                gold_scores = relevances[str(run)]

                for i in range(len(ranking[run])):
                    scores = ranking[run][i].keys()
                    CG = calculate_cg(gold_scores, scores)
                    cgs.append(CG)

                    NDCG = calculate_ndcg(gold_scores, scores)
                    ndcgs.append(NDCG)

            cg_means[missing_rate][key] = np.mean(cgs, axis=0)
            cg_deviations[missing_rate][key] = np.std(cgs, axis=0)

            ndcg_means[missing_rate][key] = np.mean(ndcgs)
            ndcg_deviations[missing_rate][key] = np.std(ndcgs)

    ndcg_means = pd.DataFrame(ndcg_means).T
    ndcg_deviations = pd.DataFrame(ndcg_deviations).T
    return (cg_means, cg_deviations), (ndcg_means, ndcg_deviations)
