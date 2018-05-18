# %%
from project.utils import DataLoader, introduce_missing_values, scale_data

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")
data = introduce_missing_values(data)
data = scale_data(data)

test_X = data.X.iloc[0:10, :]
test_y = data.y.iloc[0:10]
test = data.replace(X=test_X, y=test_y, shape=test_X.shape)

train_X = data.X.iloc[10:, :].reset_index(drop=True)
train_y = data.y.iloc[10:].reset_index(drop=True)
train = data.replace(X=train_X, y=train_y, shape=train_X.shape)

train_table = train.to_table()

# %%
from Orange.preprocess.score import ReliefF
scores = ReliefF(train_table)
for attr, score in zip(train_table.domain.attributes, scores):
    print('%.3f' % score, attr.name)

# %%
from Orange.distance import Euclidean
from Orange.data import Table
import time
t = Table.from_numpy(None, train.X)

# %%
start = time.time()
dist_model = Euclidean(normalize=True).fit(t)
print(dist_model(t))
print(time.time() - start)
