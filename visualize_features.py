# %%
from project.utils.plots import *
from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data

name = "ionosphere"
data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

mr = 0.3
data = introduce_missing_values(data, missing_rate=mr)
data = scale_data(data)

# %%
plot_nans_by_class(data)

# %%
plot_nan_percentage(data)

# %%
plot_nan_correlation(data)

# %%
plot_nan_dendogram(data)

# %%
features = ["a05", "a06"]
show_boxplots(data, features)

# %%
show_scatter_plots(data, features)

# %%
show_correlation(data, features)
