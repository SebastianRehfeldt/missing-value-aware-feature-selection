# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader
from project.utils.data_modifier import introduce_missing_values

data_loader = DataLoader()
features, labels, types = data_loader.load_data("iris", "csv")
#features = introduce_missing_values(features, types, missing_rate=0.5)

# np.unique(features["leaves"])
print(features.head())
print(labels.head())
print(types.head())
