# %%
import numpy as np
from project.utils import DataLoader, introduce_missing_values

data_loader = DataLoader("test")
features, labels, types = data_loader.load_data("csv")
#features = introduce_missing_values(features, types, missing_rate=0.5)

# np.unique(features["leaves"])
features.head()


# %%
labels.head()


# %%
types.head()

# %%
features["f2"][0] = np.nan
np.unique(features["f2"])
