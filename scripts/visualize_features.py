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
data = introduce_missing_values(data, missing_rate=mr)
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
features = ["a05", "a06"]
show_boxplots(data, features)

# %%
ax = show_scatter_plots(data, features)
ax.get_figure().savefig(os.path.join(FOLDER, "scatter.png"))

# %%
show_correlation(data, features)
