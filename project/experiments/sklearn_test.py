# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader
from project.utils.data_modifier import introduce_missing_values
from project.utils.data_scaler import scale_data

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")
#data = introduce_missing_values(data)
data = scale_data(data)
data.X.head()


# %%
import time
from project.shared.neighbors import Neighbors

start = time.time()
nn = Neighbors(data)

for i in range(data.shape[0]):
    # Get radius for k nearest neighbors within same class
    sample = data.X.iloc[i]
    #dist_cond = nn.partial_distances(sample)
    # print(dist_cond)
#print(time.time() - start)


# %%
from sklearn.metrics import pairwise_distances
from project.shared.c_distance import custom_distance

start = time.time()
kwargs = {
    "nominal_distance": 1,
    "f_types": data.f_types.values
}
distances = pairwise_distances(data.X, metric=custom_distance, **kwargs)
print(distances)
print(time.time() - start)
