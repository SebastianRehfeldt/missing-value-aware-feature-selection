# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader
from project.utils.data_modifier import introduce_missing_values
from project.utils.data_scaler import scale_data
from sklearn.preprocessing import LabelEncoder

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")
data = introduce_missing_values(data)
data = scale_data(data)

"""
y = LabelEncoder().fit_transform(data.y)
y = [str(l) for l in y]
y = pd.Series(y)
y.name = data.y.name
data = data.replace(y=y)
"""


test_X = data.X.iloc[0:10, :]
test_y = data.y.iloc[0:10]
test = data.replace(X=test_X, y=test_y, shape=test_X.shape)

train_X = data.X.iloc[10:, :].reset_index(drop=True)
train_y = data.y.iloc[10:].reset_index(drop=True)
train = data.replace(X=train_X, y=train_y, shape=train_X.shape)


# %%
train_table = train.to_table()


# %%
from Orange.preprocess.score import ReliefF
scores = ReliefF(train_table)
for attr, score in zip(train_table.domain.attributes, scores):
    print('%.3f' % score, attr.name)


# %%
from Orange.classification import TreeLearner
classifier = TreeLearner().fit_storage(train_table)
predictions = classifier(test.X.values)
for i, pred in enumerate(predictions):
    t = test.y[i].decode('utf-8')
    p = train_table.domain.class_var.values[pred]
    print("True vs Predicted: {:s} {:s}".format(t, p))


# %%
from Orange.distance import Euclidean
from Orange.data import Table
t = Table.from_numpy(None, train.X)
dist_model = Euclidean(normalize=True).fit(t)
print(t.X.shape)
dist_model(t)
