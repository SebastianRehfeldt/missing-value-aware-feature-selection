# %%
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from pprint import pprint

from project.rar.rar import RaR
from project.utils.data import DataGenerator
from project.utils.imputer import Imputer
from project.utils import DataLoader, introduce_missing_values, scale_data
from experiments.metrics import calc_ndcg

data_loader = DataLoader(ignored_attributes=["molecule_name"])
name = "heart-c"
data = data_loader.load_data(name, "arff")
data = scale_data(data)

data = introduce_missing_values(data, missing_rate=0)
print(data.shape, flush=True)

# %%
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.orange import Orange

n_runs = 1
seeds = [0] * n_runs
seeds = [42, 0, 113, 98, 234, 143, 1, 20432, 4357, 12]
missing_rates = [0.1]
missing_rates = [0.2 * i for i in range(0, 5)]
avgs = np.zeros(len(missing_rates))
stds = np.zeros(len(missing_rates))
sums = np.zeros(len(missing_rates))
data_orig = deepcopy(data)

is_synthetic = True
generator = DataGenerator(n_samples=500, n_relevant=3, n_clusters=0)

for j, mr in enumerate(missing_rates):
    print("======== {:.2f} ========".format(mr))
    ndcgs = np.zeros(n_runs)
    for i in range(n_runs):
        if is_synthetic:
            generator.set_seed(seeds[i])
            data_orig, relevance_vector = generator.create_dataset()
            imputer = Imputer(data_orig.f_types, strategy="soft")

        data_copy = deepcopy(data_orig)
        data_copy = introduce_missing_values(data_copy, mr, seed=seeds[i])
        # data_copy = imputer.complete(data_copy)

        start = time()
        selector = RaR(
            data_copy.f_types,
            data_copy.l_type,
            data_copy.shape,
            approach="deletion",
            weight_approach="imputed",
            boost_value=0,
            boost_inter=0.1,
            boost_corr=0,
            # random_state=seeds[j],
            cache_enabled=True,
            dist_method="radius",
            imputation_method="soft",
        )

        if True:
            selector = Ranking(
                data_copy.f_types,
                data_copy.l_type,
                data_copy.shape,
                eval_method="cfs",
            )

        selector.fit(data_copy.X, data_copy.y)
        pprint(selector.get_ranking())
        # print(time() - start)
        ranking = [k for k, v in selector.get_ranking()]
        ndcgs[i] = calc_ndcg(relevance_vector, ranking, False)
        print(ndcgs[i])

    avgs[j] = np.mean(ndcgs)
    stds[j] = np.std(ndcgs)
    sums[j] = np.sum([v for k, v in selector.get_ranking()])
    print(avgs[j], stds[j])

rar_results = pd.DataFrame(avgs, columns=["AVG"], index=missing_rates)
rar_results["STD"] = stds
rar_results["SUM"] = sums
rar_results = rar_results.T
rar_results.T

# %%
data_copy.X.head().T

# %%
relevance_vector.sort_values(ascending=False)

# %%
ranking

# %%
selector.get_params()

# %%
k = 4
X_new = rar.transform(data_copy.X, k)
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data_copy.replace(True, X=X_new, shape=X_new.shape, f_types=types)
X_new.corr().style.background_gradient()

# %%
from project.classifier import KNN
from project.classifier.sklearn_classifier import SKClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

knn = KNN(new_data.f_types, new_data.l_type, knn_neighbors=20)
clf = SKClassifier(data.f_types, kind="knn")
gnb = SKClassifier(data.f_types, kind="gnb")

cv = StratifiedKFold(new_data.y, n_folds=5, shuffle=True)
scorer = make_scorer(f1_score, average="micro")

scores = cross_val_score(
    knn, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
