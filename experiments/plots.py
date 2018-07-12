import os
import pandas as pd


def plot_mean_durations(FOLDER, durations):
    path = os.path.join(FOLDER, "runtimes.png")
    ax = durations.plot(kind="bar", title="Mean fitting time", rot=0)
    ax.set(xlabel="Missing Rate", ylabel="Time in seconds")
    fig = ax.get_figure()
    fig.savefig(path)


def plot_cgs(FOLDER, cgs):
    CG_FOLDER = os.path.join(FOLDER, "CG")
    os.makedirs(CG_FOLDER, exist_ok=True)

    for mr in cgs[0].keys():
        cg_means = pd.DataFrame(cgs[0][mr])
        cg_stds = pd.DataFrame(cgs[1][mr])

        ax = cg_means.plot(kind="line", title="CG over features")
        ax.set(xlabel="# Features", ylabel="CG (Mean)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_means{:s}.png").format(mr))

        ax = cg_stds.plot(kind="line", title="CG over features")
        ax.set(xlabel="# Features", ylabel="CG (Std)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_stds{:s}.png").format(mr))

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

    ax = scores[1].plot(kind="line", title=title)
    ax.set(xlabel="Missing Rate", ylabel="{:s} (Std)".format(name))
    fig = ax.get_figure()
    fig.savefig(os.path.join(FOLDER, "{:s}_deviations.png".format(name)))

    scores[0].to_csv(os.path.join(FOLDER, "{:s}_means.csv".format(name)))
    scores[1].to_csv(os.path.join(FOLDER, "{:s}_deviations.csv".format(name)))
