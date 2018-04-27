#%%
import numpy as np
from project.utils import DataLoader

"""
data_loader = DataLoader("ionosphere")
features, labels, types = data_loader.load_csv()

print(types.head())
"""


import requests
import json

url = 'https://www.openml.org/api/v1/json/data/list/data_name/madelon'
r = requests.get(url)
print(json.loads(r.content))