# %%
import numpy as np
import pandas as pd
from project.utils import DataLoader, introduce_missing_values

data_loader = DataLoader()
features, labels, types = data_loader.load_data("ionosphere", "arff")
# features = introduce_missing_values(features, types, missing_rate=0.5)

# np.unique(features["leaves"])
features.head()


# %%
labels.head()


# %%
types.head()
