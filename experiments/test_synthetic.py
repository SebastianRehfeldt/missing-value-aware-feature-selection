# %%
import pandas as pd
from time import time
from pprint import pprint

from experiments.generate_data import create_dataset
from project.utils import introduce_missing_values, scale_data

data, relevance_vector = create_dataset()
print(data.shape, flush=True)

data = introduce_missing_values(data, missing_rate=0)
data = scale_data(data)
relevance_vector

# %%
from project.rar.rar import RaR

start = time()
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=1,
    approach="deletion",
    use_pearson=False,
    n_targets=0,
    n_subspaces=1000,
    contrast_iterations=200,
)

rar.fit(data.X, data.y)
pprint(rar.get_ranking())
print(time() - start)

# %%
X_new = rar.transform(data.X, 5)
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data.replace(True, X=X_new, shape=X_new.shape, f_types=types)
new_data.X.shape

# %%
import numpy as np
from project.classifier import KNN
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

knn = KNN(new_data.f_types, new_data.l_type, knn_neighbors=20)
clf = KNeighborsClassifier(n_neighbors=20)
gnb = GaussianNB()

cv = StratifiedKFold(new_data.y, n_folds=3, shuffle=True)
scorer = make_scorer(f1_score, average="micro")

scores = cross_val_score(
    clf, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
