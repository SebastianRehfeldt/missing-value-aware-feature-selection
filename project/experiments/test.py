#%%
from project.utils import load_data_from_uci

features, labels, types = load_data_from_uci("artificial-characters")

print(features.head())
print(labels.head())
print(types)