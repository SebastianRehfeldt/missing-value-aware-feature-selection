# %%
import os
import json
import requests
import numpy as np
import pandas as pd

from project import DATA_PATH

FOLDER = os.path.join(DATA_PATH, "statistics")

try:
    os.makedirs(FOLDER)
    base = 'https://www.openml.org/api/v1/json/data/list/'
    raw_data = json.loads(requests.get(base).content)
    with open(os.path.join(FOLDER, "raw_data.json"), 'w') as outfile:
        json.dump(raw_data, outfile, indent=4)

except:
    with open(os.path.join(FOLDER, "raw_data.json")) as infile:
        raw_data = json.load(infile)["data"]["dataset"]
        for d in raw_data:
            metrics = d["quality"]
            new_metrics = {x["name"]: x["value"] for x in metrics}
            d["quality"] = new_metrics

# %%
stats = {}
stats["n_datasets"] = len(raw_data)

# %%
raw_data

# DONE
# SIZE (instances, features)
# MR BY DIMENSIONS

# MIXED, NUMERICAL, NOMINAL FEATURES (PIE)
# OVERVIEW TABLE FOR SPECIFIC DATASETS (instances, features, classes, majority/n, mr)
# MAX NOMINAL ATTR DISTINCT VALUES (HIST)
# MISSING RATES (HIST)
# INSTANCES WITH MISSING VALUES (HIST, COMBINED WITH MRS AS SCATTER)
# DATASETS WITH MISSING VALUES (PIE)
# QUERY FOR FINDING DATASETS (size, mr, type)


# %%
def plot_mr_by_dimensions(data, use_log=True, max_f=1e2, max_n=1e4, bin=True):
    n = len(data)
    dimensions = [None] * n
    missing_rates = np.zeros(n)

    show_in_plot = np.zeros(n, dtype=bool)
    if use_log:
        show_in_plot = np.ones(n, dtype=bool)

    for i, d in enumerate(data):
        metrics = d["quality"]
        dimensions[i] = {
            "samples": float(metrics["NumberOfInstances"]),
            "features": float(metrics["NumberOfFeatures"]),
        }

        n_samples, n_features = list(dimensions[i].values())
        if not use_log and n_features < max_f and n_samples < max_n:
            show_in_plot[i] = True

        n_values = n_samples * n_features
        mr = float(metrics["NumberOfMissingValues"]) / n_values
        missing_rates[i] = mr

    colors = missing_rates[show_in_plot]
    if bin:
        colors = colors > 0

    sorted_indices = np.argsort(colors)
    colors = colors[sorted_indices]
    dimensions = pd.DataFrame(dimensions)

    df = dimensions[show_in_plot].iloc[sorted_indices]
    ax = df.plot.scatter("samples", "features", c=colors, cmap='coolwarm')

    if use_log:
        ax.set_yscale('log')
        ax.set_xscale('log')

    return ax, (dimensions, missing_rates)


fig, data = plot_mr_by_dimensions(raw_data, False, bin=False)
dimensions, missing_rates = data

# %%
colors