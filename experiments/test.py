# %%
import pandas as pd
from time import time
from pprint import pprint

from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data
# from project.utils.imputer import Imputer

data_loader = DataLoader(ignored_attributes=["molecule_name"])
name = "madelon"
name = "boston"
name = "analcatdata_reviewer"
name = "credit-approval"  # standard config
name = "musk"  # standard config
name = "isolet"
name = "semeion"
name = "ionosphere"  # 800 subspaces, alpha=0.02, 250 iterations ,(1,3)
name = "iris"
name = "heart-c"  # 800 subspaces, alpha = 0,2, 100 iterations, (1,3)
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

data = introduce_missing_values(data, missing_rate=0.4)
data = scale_data(data)
data.X.head()

# %%
from project.rar.rar import RaR

start = time()
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=1,
    approach="deletion",
    n_targets=1,
    n_subspaces=800,
    subspace_size=(1, 3),
    contrast_iterations=100,
    alpha=0.2,
    redundancy_approach="arvind",
    sample_slices=False,
)

rar.fit(data.X, data.y)
pprint(rar.get_ranking())
print(time() - start)

# %%
k = 20
X_new = rar.transform(data.X, k)
X_new.head()
X_new.corr().style.background_gradient()

# %%
if True:
    types = pd.Series(data.f_types, X_new.columns.values)
    X_new = Imputer(types, strategy="knn")._complete(X_new)
    X_new.head()

# %%
from project.feature_selection import Filter
selector = Filter(data.f_types, data.l_type, data.shape).fit(data.X, data.y)
selector.get_ranking()

# %%
from project.feature_selection import RKNN
selector = RKNN(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=4,
    # n_subspaces=100,
).fit(data.X, data.y)
selector.get_ranking()

# %%
X_new = selector.transform(data.X, k)
X_new.corr().style.background_gradient()

# %%
if True:
    types = pd.Series(data.f_types, X_new.columns.values)
    X_new = Imputer(types, strategy="knn")._complete(X_new)

# %%
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data.replace(True, X=X_new, shape=X_new.shape, f_types=types)

print(new_data.X.shape)

import numpy as np
from project.classifier import KNN
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

knn = KNN(new_data.f_types, new_data.l_type, knn_neighbors=20)
clf = KNeighborsClassifier(n_neighbors=20)
gnb = GaussianNB()

cv = StratifiedKFold(new_data.y, n_folds=5, shuffle=True)
scorer = make_scorer(f1_score, average="micro")

scores = cross_val_score(
    clf, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
