# %%
import os
from project import DATA_PATH
from scripts.utils import *

FOLDER = os.path.join(DATA_PATH, "statistics")
PLOT_FOLDER = os.path.join(FOLDER, "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)
raw_data = get_raw_data(FOLDER)

# %%
print_dataset(raw_data, ["iris"])

# %%
find(
    raw_data,
    f=(13, 50),
    s=(100, 1000),
    c=(2, 10),
    mr=(30, 100),
)

# %%
# PLOT MR BY DIMENSIONS
fig, data = plot_mr_by_dimensions(raw_data, False, bin_=False)
fig.savefig(os.path.join(PLOT_FOLDER, "mr_by_dimensions.png"))
dimensions, missing_rates = data

# %%
# GLOBAL STATISTICS TABLE
summary, types = get_global_statistics(raw_data, missing_rates)
summary.to_csv(os.path.join(PLOT_FOLDER, "summary_global.csv"))
summary

# %%
# TABLES AND HIST FOR INCOMPLETE DATASETS
mr = 0
name = "summary_incomplete_{:.2f}.csv".format(mr)
summary_mr, ratios = get_statistics_for_mr(raw_data, missing_rates, mr)
summary_mr.to_csv(os.path.join(PLOT_FOLDER, name))

fig = plot_ratios(ratios)
name = "ratios_of_incomplete_instances_{:.2f}.png".format(mr)
fig.savefig(os.path.join(PLOT_FOLDER, name))
summary_mr

# %%
# MR DISTRIBUTION BY TYPE + PERCENTAGE OF INCOMPLETE DATASETS
fig, summary_types = plot_by_type(missing_rates, types)
fig.savefig(os.path.join(PLOT_FOLDER, "mr_by_type.png"))
summary_types.to_csv(os.path.join(PLOT_FOLDER, "summary_types.csv"))
summary_types

# %%
names = [
    "heart-c", "ionosphere", "semeion", "madelon", "musk", "isolet",
    "hepatitis", "vote", "soybean", "anneal", "schizo"
]
overview = get_overview(raw_data, types, missing_rates, names)
overview.to_csv(os.path.join(PLOT_FOLDER, "summary_datasets.csv"))
overview.T
