# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")

"""
print(data.features.head())
print(data.labels.head())
print(data.types.head())
"""


# %%
from project.utils.data_modifier import introduce_missing_values

data = introduce_missing_values(data)
data.features.head()


# %%
from project.utils.data_scaler import scale_data
data = scale_data(data)
data.features.head()


# %%
from project.randomKNN.random_knn import RKNN
rknn = RKNN(data, method="imputation")
rknn.fit_transform().head()


# %%
rknn.ranking
