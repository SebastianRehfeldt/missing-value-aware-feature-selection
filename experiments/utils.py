import os
import json
import numpy as np
import pandas as pd


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


def write_results(FOLDER, relevances, durations, rankings):
    relevances.to_csv(os.path.join(FOLDER, "relevances.csv"))
    path = os.path.join(FOLDER, "runtimes.json")
    write_json(durations, path)
    path = os.path.join(FOLDER, "rankings.json")
    write_json(rankings, path)


def read_results(FOLDER):
    path = os.path.join(FOLDER, "relevances.csv")
    relevances = pd.DataFrame.from_csv(path)
    path = os.path.join(FOLDER, "runtimes.json")
    durations = read_json(path)
    path = os.path.join(FOLDER, "rankings.json")
    rankings = read_json(path)
    return relevances, durations, rankings


def get_mean_durations(durations):
    mean_durations = {}
    for missing_rate in durations.keys():
        mean_durations[missing_rate] = {}

        for key in durations[missing_rate].keys():
            mean_time = np.mean(durations[missing_rate][key], axis=1)
            mean_time = np.mean(mean_time)
            mean_durations[missing_rate][key] = mean_time

    return pd.DataFrame(mean_durations).T
