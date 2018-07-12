# %%
import numpy as np
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
name = "iris"
name = "heart-c"  # 800 subspaces, alpha = 0,2, 100 iterations, (1,3)
name = "isolet"
name = "semeion"
name = "ionosphere"  # 800 subspaces, alpha=0.02, 250 iterations ,(1,3)
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

mr = 0.5
data = introduce_missing_values(data, missing_rate=mr)
data = scale_data(data)

# %%
from project.rar.rar import RaR

start = time()
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=1,
    approach="fuzzy",
    n_targets=1,
    n_subspaces=800,
    subspace_size=(1, 3),
    contrast_iterations=250,
    alpha=0.02,
    redundancy_approach="tom",
    weight=min(0.9, (1 - mr)**2),
    random_state=42,
    cache_enabled=True,
    sample_slices=True,
)

rar.fit(data.X, data.y)
pprint(rar.get_ranking())
print(time() - start)

# %%
k = 5
X_new = rar.transform(data.X, k)
X_new.head()
X_new.corr().style.background_gradient()

# %%
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data.replace(True, X=X_new, shape=X_new.shape, f_types=types)

print(new_data.X.shape)

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
    knn, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
