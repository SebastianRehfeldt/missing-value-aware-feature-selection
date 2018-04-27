# %%
import numpy as np
from project.utils import DataLoader, introduce_missing_values

data_loader = DataLoader("musk")
features, labels, types = data_loader.load_data()
features = introduce_missing_values(features, missing_rate=0.1)

features.head()
