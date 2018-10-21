import os
import json
import requests
import numpy as np
import pandas as pd
from pprint import pprint


def get_raw_data(FOLDER):
    try:
        os.makedirs(FOLDER)
        os.makedirs(os.path.join(FOLDER, "plots"))
        base = 'https://www.openml.org/api/v1/json/data/list/'
        raw_data = json.loads(requests.get(base).content)
        raw_data = raw_data["data"]["dataset"]

        data = []
        for d in raw_data:
            if d["quality"]:
                data.append(d)

        with open(os.path.join(FOLDER, "raw_data.json"), 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except:
        path = os.path.join(FOLDER, "raw_data.json")
        with open(path) as infile:
            data = json.load(infile)["data"]["dataset"]
    finally:
        for d in data:
            metrics = d["quality"]
            new_metrics = {x["name"]: float(x["value"]) for x in metrics}
            d["quality"] = new_metrics
        return data


def print_dataset(data, names):
    for d in data:
        if d["name"] in names:
            pprint(d)


def find(data, f=(0, 1e10), s=(0, 1e10), c=(2, 1e10), mr=(0, 1), types=[]):
    ids, names = [], []
    for d in data:
        m = d["quality"]
        n_values = m["NumberOfFeatures"] * m["NumberOfInstances"]
        missing_rate = m["NumberOfMissingValues"] / n_values * 100

        t = "numeric"
        if d["quality"]["NumberOfSymbolicFeatures"] > 1:
            if d["quality"]["NumberOfNumericFeatures"] > 0:
                t = "mixed"
            else:
                t = "nominal"

        if not f[0] <= m["NumberOfFeatures"] - 1 <= f[1]:
            continue
        if not s[0] <= m["NumberOfInstances"] <= s[1]:
            continue
        if not c[0] <= m["NumberOfClasses"] <= c[1]:
            continue
        if not mr[0] <= missing_rate <= mr[1]:
            continue
        if len(types) != 0 and t not in types:
            continue

        ids.append(d["did"])
        names.append(d["name"])
    return ids, names


def plot_mr_by_dimensions(data, use_log=True, max_f=1e2, max_n=1e4, bin_=True):
    # PREPARATION FOR LOOP
    n = len(data)
    dimensions = [None] * n
    mr_s = np.zeros(n)
    show = np.ones(n, dtype=bool) if use_log else np.zeros(n, dtype=bool)

    # COMBINE RESULTS INTO ARRAY
    for i, d in enumerate(data):
        dimensions[i] = {
            "Samples": d["quality"]["NumberOfInstances"],
            "Features": d["quality"]["NumberOfFeatures"],
        }

        n_samples, n_features = list(dimensions[i].values())
        if not use_log and n_features < max_f and n_samples < max_n:
            show[i] = True

        n_values = n_samples * n_features
        mr_s[i] = d["quality"]["NumberOfMissingValues"] / n_values

    # CREATE COLORS AND SORT BY MR (MR BUBBLES IN FRONT)
    colors = mr_s[show] > 0 if bin_ else mr_s[show]
    sorted_indices = np.argsort(colors)

    # PLOT
    dimensions = pd.DataFrame(dimensions)
    df = dimensions[show].iloc[sorted_indices]
    colors = colors[sorted_indices]
    ax = df.plot.scatter(
        "Samples",
        "Features",
        c=colors,
        cmap='gist_heat_r',
        sharex=False,
        edgecolors="gray",
        linewidths=0.25,
    )

    if use_log:
        ax.set_yscale('log')
        ax.set_xscale('log')

    return ax.get_figure(), (dimensions, mr_s)


def get_global_statistics(data, missing_rates, imbalance_ratio=10):
    n = len(data)
    types, bin_count, imbalanced_count = [None] * n, 0, 0

    for i, d in enumerate(data):
        # GET TYPE OF DATASET
        try:
            if d["quality"]["NumberOfSymbolicFeatures"] > 1:
                if d["quality"]["NumberOfNumericFeatures"] > 0:
                    types[i] = "mixed"
                else:
                    types[i] = "nominal"
            else:
                types[i] = "numeric"

            # INCREASE BINARY AND IMBALANCED COUNT
            if d["quality"]["NumberOfClasses"] == 2:
                bin_count += 1

            s_min = d["quality"]["MinorityClassSize"]
            s_max = d["quality"]["MajorityClassSize"]
            if s_min * imbalance_ratio < s_max:
                imbalanced_count += 1
        except:
            pass

    data = np.zeros((2, 6))
    row_names = ["COUNT", "PERCENTAGE"]
    col_names = [
        "MIXED",
        "NOMINAL",
        "NUMERIC",
        "BINARY",
        "IMBALANCED",
        "INCOMPLETE",
    ]
    summary = pd.DataFrame(data, columns=col_names, index=row_names)
    summary["MIXED"] = types.count("mixed")
    summary["NOMINAL"] = types.count("nominal")
    summary["NUMERIC"] = types.count("numeric")
    summary["BINARY"] = bin_count
    summary["IMBALANCED"] = imbalanced_count
    summary["INCOMPLETE"] = np.sum(missing_rates > 0)
    summary.iloc[1] = summary.iloc[0] / n * 100
    decimals = pd.Series([0, 2], index=['COUNT', 'PERCENTAGE'])
    return summary.T.round(decimals), np.asarray(types)


def get_statistics_for_mr(data, missing_rates, thresh=0):
    n = len(data)
    ratios = np.zeros(n)
    n_samples = np.zeros(n)
    n_features = np.zeros(n)

    for i, d in enumerate(data):
        m = d["quality"]
        if m["NumberOfMissingValues"] == 0:
            continue

        n_samples[i] = m["NumberOfInstances"]
        n_features[i] = m["NumberOfFeatures"]
        ratios[i] = m["NumberOfInstancesWithMissingValues"] / n_samples[i]

    has_nan = missing_rates > thresh
    m = missing_rates[has_nan] * 100
    r = ratios[has_nan] * 100
    s = n_samples[has_nan]
    f = n_features[has_nan]

    data = np.zeros((4, 4))
    row_names = ["MAX", "MEDIAN", "MEAN", "STD"]
    col_names = [
        "MISSING_RATE",
        "RATIO_INCOMPLETE_SAMPLES",
        "N_FEATURES",
        "N_SAMPLES",
    ]
    summary = pd.DataFrame(data, columns=col_names, index=row_names)
    summary["MISSING_RATE"] = [np.max(m), np.median(m), np.mean(m), np.std(m)]
    summary["RATIO_INCOMPLETE_SAMPLES"] = [
        np.max(r), np.median(r),
        np.mean(r), np.std(r)
    ]
    summary["N_SAMPLES"] = [np.max(s), np.median(s), np.mean(s), np.std(s)]
    summary["N_FEATURES"] = [np.max(f), np.median(f), np.mean(f), np.std(f)]
    return summary.T.round(2), r


def plot_ratios(ratios):
    x = "Percentage of incomplete samples"
    weights = np.ones_like(ratios) / len(ratios)
    ax = pd.DataFrame(ratios).plot.hist(
        bins=20,
        weights=weights,
        edgecolor="black",
        linewidth=1,
        range=(0, 100))
    ax.set(xlabel=x, ylabel="Relative Frequency")
    ax.legend_.remove()
    return ax.get_figure()


def plot_by_type(missing_rates, types):
    has_nan = missing_rates > 0

    mixed_count_complete = (types == "mixed").sum()
    mixed_count_incomplete = (types[has_nan] == "mixed").sum()
    mixed = mixed_count_incomplete / mixed_count_complete

    nominal_count_complete = (types == "nominal").sum()
    nominal_count_incomplete = (types[has_nan] == "nominal").sum()
    nominal = nominal_count_incomplete / nominal_count_complete

    numeric_count_complete = (types == "numeric").sum()
    numeric_count_incomplete = (types[has_nan] == "numeric").sum()
    numeric = numeric_count_incomplete / numeric_count_complete

    summary = pd.DataFrame(data=[mixed, nominal, numeric])
    summary.index = ["MIXED", "NOMINAL", "NUMERIC"]
    summary.columns = ["PERCENTAGE"]
    summary *= 100

    data = np.zeros((has_nan.sum(), 2))
    df = pd.DataFrame(data, columns=["MISSING_RATE", "TYPES"])
    df["MISSING_RATE"] = missing_rates[missing_rates > 0]
    df["TYPES"] = types[missing_rates > 0]
    ax = df.boxplot(grid=False, by="TYPES")
    ax.set(xlabel="", ylabel="Missing Rate", title="Missing rate by type")
    for t in ["mixed", "nominal", "numeric"]:
        print(t, float(df[df["TYPES"] == t].mean()))
    fig = ax.get_figure()
    fig.suptitle('')
    return fig, summary.round(2)


def get_overview(data, types, missing_rates, names):
    df = pd.DataFrame()
    for i, d in enumerate(data):
        if d["name"] in names:
            m = d["quality"]
            class_ratio = m["MajorityClassSize"] / m["NumberOfInstances"] * 100
            n_incomplete = m["NumberOfInstancesWithMissingValues"]
            na_ratio = n_incomplete / m["NumberOfInstances"] * 100

            if d["name"] == "anneal" and m["NumberOfClasses"] != 5:
                continue
            if d["name"] == "soybean" and m["NumberOfClasses"] != 19:
                continue

            stats = []
            stats.append(m["NumberOfInstances"])
            stats.append(m["NumberOfFeatures"] - 1)
            stats.append(m["NumberOfClasses"])
            stats.append(np.round(class_ratio, 2))
            stats.append(np.round(missing_rates[i] * 100, 2))
            stats.append(np.round(na_ratio, 2))
            stats.append(types[i])
            df[d["name"]] = stats

    df.index = [
        "INSTANCES", "FEATURES", "CLASSES", "MAJORITY RATIO", "MISSING RATE",
        "INCOMPLETE SAMPLES (%)", "DATASET TYPE"
    ]
    return df
