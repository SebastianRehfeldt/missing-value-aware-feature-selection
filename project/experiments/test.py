# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
features, labels, types = data_loader.load_data("iris", "arff")

print(features.head())
"""
print(labels.head())
print(types.head())
"""


# %%
from project.utils.data_modifier import introduce_missing_values

features = introduce_missing_values(features, types)
features.head()


# %%
from project.utils.data_scaler import scale_data
data = scale_data(features, types)
data.head()


# %%
from project.randomKNN.random_knn import RKNN
rknn = RKNN(features, labels, types, method="imputation")
rknn.fit_transform()
