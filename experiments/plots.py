import os
import pandas as pd


def plot_mean_durations(durations, FOLDER):
    path = os.path.join(FOLDER, "runtimes.png")
    ax = durations.plot(kind="bar", title="Mean fitting time", rot=0)
    ax.set(xlabel="Missing Rate", ylabel="Time in seconds")
    fig = ax.get_figure()
    fig.savefig(path)


def plot_ndcgs(ndcgs, FOLDER):
    NDCG_FOLDER = os.path.join(FOLDER, "NDCG")
    os.makedirs(NDCG_FOLDER, exist_ok=True)

    # MEAN NDCG
    ax = ndcgs[0].plot(kind="line", title="NDCG over Missing Rate")
    ax.set(xlabel="Missing Rate", ylabel="NDCG (Mean)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(NDCG_FOLDER, "ndcg_means.png"))

    # STD NDCG
    ax = ndcgs[1].plot(kind="line", title="NDCG over Missing Rate")
    ax.set(xlabel="Missing Rate", ylabel="NDCG (Std)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(NDCG_FOLDER, "ndcg_deviations.png"))

    ndcgs[0].to_csv(os.path.join(NDCG_FOLDER, "ndcg_means.csv"))
    ndcgs[1].to_csv(os.path.join(NDCG_FOLDER, "ndcg_deviations.csv"))


def plot_cgs(cgs, FOLDER):
    CG_FOLDER = os.path.join(FOLDER, "CG")
    os.makedirs(CG_FOLDER, exist_ok=True)

    for mr in cgs[0].keys():
        cg_means = pd.DataFrame(cgs[0][mr])
        cg_stds = pd.DataFrame(cgs[1][mr])

        ax = cg_means.plot(kind="line", title="Cumulative Gain over features")
        ax.set(xlabel="# Features", ylabel="Cumulative Gain (Mean)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_means{:s}.png").format(mr))

        ax = cg_stds.plot(kind="line", title="Cumulative Gain over features")
        ax.set(xlabel="# Features", ylabel="Cumulative Gain (Std)")
        fig = ax.get_figure()
        fig.savefig(os.path.join(CG_FOLDER, "cg_stds{:s}.png").format(mr))

        # STATS
        cg_means.to_csv(os.path.join(CG_FOLDER, "cg_means{:s}.csv").format(mr))
        cg_stds.to_csv(os.path.join(CG_FOLDER, "cg_stds{:s}.csv").format(mr))
