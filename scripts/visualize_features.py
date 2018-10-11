# %%
import os
from project import DATA_PATH
from project.utils.plots import *
from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data

name = "ionosphere"
FOLDER = os.path.join(DATA_PATH, "plots", name)
os.makedirs(FOLDER, exist_ok=True)

data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

mr = 0.3
data = introduce_missing_values(
    data, missing_rate=mr, missing_type="predictive")
data = scale_data(data)

# %%
ax = plot_nans_by_class(data)
ax.get_figure().savefig(os.path.join(FOLDER, "nans_by_class.png"))

# %%
ax = plot_nan_percentage(data)
ax.get_figure().savefig(os.path.join(FOLDER, "nan_percentage.png"))

# %%
ax = plot_nan_correlation(data)
ax.get_figure().savefig(os.path.join(FOLDER, "nan_correlation.png"))

# %%
ax = plot_nan_dendogram(data)
ax.get_figure().savefig(os.path.join(FOLDER, "nan_dendogram.png"))

# %%
import matplotlib.pyplot as plt


def show_boxplots(data, features=None):
    features = data.X.columns if features is None else features
    df = data.X[features]
    df["class"] = data.y
    ax = df.boxplot(grid=False, by="class")
    fig = ax.get_figure()
    fig.suptitle("")
    plt.title('')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Standardized value', fontsize=14)
    return fig


features = ["a06"]
fig = show_boxplots(data, features)
fig.savefig(os.path.join(FOLDER, "boxplot.pdf"), bbox_inches="tight")

# %%
ax = show_scatter_plots(data, features)
ax.get_figure().savefig(os.path.join(FOLDER, "scatter.png"))

# %%
show_correlation(data, features)
