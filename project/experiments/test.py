# %%
import numpy as np
from project.utils import DataLoader

data_loader = DataLoader("musk")
features, labels, types = data_loader.load_data()

print(types.head())
