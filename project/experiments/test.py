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
# rknn.fit_transform().head()


# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from project.randomKNN.random_knn import RKNN
from project.randomKNN.knn import KNN

rknn = RKNN(data, method="imputation")
knn = KNN(data.types)
y = pd.Series(LabelEncoder().fit_transform(data.labels))
cv = StratifiedKFold(y, n_folds=5, shuffle=True)

pipe1 = Pipeline(steps=[
    ('reduce', rknn),
    ('classify', knn)
])
score1 = cross_val_score(pipe1, data.features, y, cv=cv,
                         scoring="accuracy", n_jobs=-1)

print("score1", score1, flush=True)


# %%
pipe2 = Pipeline(steps=[
    ("imputer", Imputer(strategy="mean", axis=0)),
    ('classify', knn)
])
score2 = cross_val_score(pipe2, data.features, y, cv=cv,
                         scoring="accuracy", n_jobs=-1)
print("score2", score2, flush=True)


# %%
pipe3 = Pipeline(steps=[
    ('classify', knn)
])

score3 = cross_val_score(pipe3, data.features, y, cv=cv,
                         scoring="accuracy", n_jobs=-1)
print("score3", score3, flush=True)
