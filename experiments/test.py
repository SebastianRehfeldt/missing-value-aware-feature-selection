# %%
import pandas as pd
from time import time
from pprint import pprint

from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data


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
