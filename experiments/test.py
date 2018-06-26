# %%
import pandas as pd
from time import time
from pprint import pprint

from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data

<<<<<<< HEAD
data_loader = DataLoader()
name = "credit-approval"
name = "madelon"
name = "boston"
name = "analcatdata_reviewer"
name = "iris"
name = "musk"
name = "heart-c"
name = "isolet"
name = "ionosphere"
name = "semeion"
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

data = introduce_missing_values(data, missing_rate=0)
data = scale_data(data)
data.X.head()

# %%
"""
from project.utils.imputer import Imputer
imputer = Imputer(data.f_types, strategy="mice")
completed = imputer.complete(data)

data.X.head()
"""

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
    n_targets=1,
    n_subspaces=800,
    subspace_size=(1, 3),
    slicing_method="simple",
)

rar.fit(data.X, data.y)
pprint(rar.get_ranking())
print(time() - start)

# %%
X_new = rar.transform(data.X, 3)
X_new.head()
X_new.corr().style.background_gradient()

# %%
from project.feature_selection import Filter
filter_ = Filter(data.f_types, data.l_type, data.shape).fit(data.X, data.y)
filter_.get_ranking()

# %%
from project.feature_selection import RKNN
selector = RKNN(
    data.f_types,
    data.l_type,
    data.shape,
    n_jobs=3,
    n_subspaces=600,
).fit(data.X, data.y)
selector.get_ranking()

# %%
selector.get_ranking()
# %%
X_new = selector.transform(data.X, 3)
X_new.corr().style.background_gradient()

# %%
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

cv = StratifiedKFold(new_data.y, n_folds=5, shuffle=True)
scorer = make_scorer(f1_score, average="micro")

scores = cross_val_score(
    knn, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
=======

if __name__ == "__main__":


    data_loader = DataLoader()
    name = "credit-approval"
    name = "madelon"
    name = "boston"
    name = "analcatdata_reviewer"
    name = "musk"
    name = "semeion"
    name = "isolet"
    name = "ionosphere"
    name = "heart-c"
    name = "iris"
    data = data_loader.load_data(name, "arff")
    print(data.shape, flush=True)

    data = introduce_missing_values(data, missing_rate=0.0)
    data = scale_data(data)
    data.X.head()

    # %%
    """
    from project.utils.imputer import Imputer
    imputer = Imputer(data.f_types, strategy="mice")
    completed = imputer.complete(data)
    
    data.X.head()
    """

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
        max_subspaces=5000,
        contrast_iterations=100,
    )

    rar.fit(data.X, data.y)
    pprint(rar.get_ranking())
    print(time() - start)

    # %%
    X_new = rar.transform(data.X, 5)

    X_new.head()
    X_new.corr().style.background_gradient()

    # %%
    # rar.redundancies["V193"]
    # rar.get_ranking()

    # %%
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
        knn, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
    print(np.mean(scores), scores)
>>>>>>> fd01c4c2de422c878d53fa8bf584308bb9381d5f
