# %%
from time import time
from project.utils import DataLoader, introduce_missing_values, scale_data

data_loader = DataLoader()
data = data_loader.load_data("isolet", "arff")
data = introduce_missing_values(data, missing_rate=0)
data = scale_data(data)

t = data.to_table()
t

# %%
from Orange.preprocess.score import ReliefF
start = time()
scores = ReliefF(t)
for attr, score in zip(t.domain.attributes, scores):
    print('%.3f' % score, attr.name)
print("Time: ", time() - start)

# %%
from Orange.distance import Euclidean

start = time()
dist_model = Euclidean(normalize=True).fit(t)
print(dist_model(t))
print(time() - start)

# %%
from Orange.classification import RandomForestLearner
learner = RandomForestLearner()
learner.score_data(t)