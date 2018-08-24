import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_durations(FOLDER, durations):
    path = os.path.join(FOLDER, "runtimes.png")

    kind = "bar" if durations.shape[0] <= 2 else "line"
    ax = durations.plot(kind=kind, title="Mean fitting time", rot=0)
    ax.set(xlabel="Missing Rate", ylabel="Time in seconds")
    fig = ax.get_figure()
    fig.savefig(path)
    plt.close(fig)


def plot_cgs(FOLDER, cgs, name):
    CG_FOLDER = os.path.join(FOLDER, name)
    os.makedirs(CG_FOLDER, exist_ok=True)

    for mr in cgs[0].keys():
        cg_means = pd.DataFrame(cgs[0][mr])
        cg_stds = pd.DataFrame(cgs[1][mr])
        cg_means.loc[-1] = np.zeros(cg_means.shape[1])
        cg_stds.loc[-1] = np.zeros(cg_stds.shape[1])
        cg_means.index += 1
        cg_stds.index += 1
        cg_means.sort_index(inplace=True)
        cg_stds.sort_index(inplace=True)

        ax = cg_means.plot(kind="line", title="CG over features")
        ax.set(xlabel="# Features", ylabel="CG (Mean)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_means{:s}.png").format(mr))
        plt.close(fig)

        ax = cg_stds.plot(kind="line", title="CG over features")
        ax.set(xlabel="# Features", ylabel="CG (Std)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_stds{:s}.png").format(mr))
        plt.close(fig)

        # STATS
        cg_means.to_csv(os.path.join(CG_FOLDER, "cg_means{:s}.csv").format(mr))
        cg_stds.to_csv(os.path.join(CG_FOLDER, "cg_stds{:s}.csv").format(mr))


def plot_scores(folder_, scores, name):
    FOLDER = os.path.join(folder_, name)
    os.makedirs(FOLDER, exist_ok=True)

    title = "{:s} over Missing Rate".format(name)

    ax = scores[0].plot(kind="line", title=title)
    ax.set(xlabel="Missing Rate", ylabel="{:s} (Mean)".format(name))
    fig = ax.get_figure()
    fig.savefig(os.path.join(FOLDER, "{:s}_means.png".format(name)))
    plt.close(fig)

    ax = scores[1].plot(kind="line", title=title)
    ax.set(xlabel="Missing Rate", ylabel="{:s} (Std)".format(name))
    fig = ax.get_figure()
    fig.savefig(os.path.join(FOLDER, "{:s}_deviations.png".format(name)))
    plt.close(fig)

    scores[0].to_csv(os.path.join(FOLDER, "{:s}_means.csv".format(name)))
    scores[1].to_csv(os.path.join(FOLDER, "{:s}_deviations.csv".format(name)))


def plot_aucs(folder, aucs, metric="CG", name=""):
    FOLDER = os.path.join(folder, "AUC")
    os.makedirs(FOLDER, exist_ok=True)

    title = "AUC of {:s} over Missing Rate".format("CG")
    ax = aucs.plot(kind="bar", title=title)
    ax.set(ylabel="AUC")

    fig = ax.get_figure()
    fig.savefig(os.path.join(FOLDER, "aucs{:s}.png".format(name)))
    plt.close(fig)
    aucs.to_csv(os.path.join(FOLDER, "aucs{:s}.csv".format(name)))
