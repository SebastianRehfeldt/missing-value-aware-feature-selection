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
from experiments.metrics import calc_ndcg, calc_cg

data_loader = DataLoader(ignored_attributes=["molecule_name"])
name = "ionosphere"
data = data_loader.load_data(name, "arff")
data = scale_data(data)

data = introduce_missing_values(data, missing_rate=0)
print(data.shape, flush=True)

# %%
from project.feature_selection import Filter, RKNN, SFS
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.orange import Orange
from project.feature_selection.baseline import Baseline

n_runs = 5
seeds = [42] * n_runs
seeds = [42, 0, 113, 98, 234, 143, 1, 20432, 4357, 12]

np.random.seed(1)
seeds = np.random.randint(0, 1000, n_runs)
#[ 37 235 908  72 767]

missing_rates = [0.5]
missing_rates = [0.2 * i for i in range(0, 5)]
cgs = np.zeros(len(missing_rates))
avgs = np.zeros(len(missing_rates))
stds = np.zeros(len(missing_rates))
sums = np.zeros(len(missing_rates))
data_orig = deepcopy(data)

is_synthetic = True
generator = DataGenerator(
    n_samples=500, n_relevant=2, n_clusters=2, n_discrete=10)
shuffle_seed = 0

for j, mr in enumerate(missing_rates):
    print("======== {:.2f} ========".format(mr))
    ndcgs = np.zeros(n_runs)
    cgs_run = np.zeros(n_runs)
    for i in range(n_runs):
        if is_synthetic:
            generator.set_seed(seeds[i])
            data_orig, relevance_vector = generator.create_dataset()
        else:
            data_orig = data

        data_copy = deepcopy(data_orig)
        imputer = Imputer(data_orig.f_types, strategy="soft")
        data_copy = introduce_missing_values(data_copy, mr, seed=seeds[i])
        # data_copy = imputer.complete(data_copy)

        data_copy.shuffle_columns(seed=shuffle_seed)
        shuffle_seed += 1

        start = time()
        selector = RaR(
            data_copy.f_types,
            data_copy.l_type,
            data_copy.shape,
            alpha=0.02,  # * (1 + mr),
            approach="fuzzy",
            weight_approach="multiple",
            boost_value=0.1,
            boost_inter=0,
            boost_corr=0,
            regularization=1,
            weight=1,
            n_targets=1,
            # random_state=seeds[j],
            cache_enabled=True,
            dist_method="distance",
            imputation_method="mice",
            subspace_size=(1, 2),
            active_sampling=False,
        )

        if False:
            selector = Ranking(
                data_copy.f_types,
                data_copy.l_type,
                data_copy.shape,
                eval_method="mrmr",
            )

        X = data_copy.X
        #X = X.round(2)

        selector.fit(X, data_copy.y)
        # pprint(selector.get_ranking())
        # print(time() - start)
        ranking = [k for k, v in selector.get_ranking()]

        if is_synthetic:
            ndcgs[i] = calc_ndcg(relevance_vector, ranking, False)
            n_relevant = np.count_nonzero(relevance_vector.values)
            cgs_run[i] = calc_cg(relevance_vector, ranking)[n_relevant]
            print(cgs_run[i])
        else:
            print(selector.get_ranking())

        #print(selector.hics.evaluate_subspace(["f7"])[0])
        #print(selector.hics.evaluate_subspace(["f7", "f11"])[0])

    cgs[j] = np.mean(cgs_run)
    stds[j] = np.std(cgs_run)
    avgs[j] = np.mean(ndcgs)
    sums[j] = np.sum([v for k, v in selector.get_ranking()])
    print(cgs[j], stds[j])

rar_results = pd.DataFrame(cgs, columns=["CG"], index=missing_rates)
rar_results["STD"] = stds
rar_results["NDCG"] = avgs
rar_results["SUM"] = sums
rar_results = rar_results.T
rar_results.T

# %%
for col in X:
    print(type(X.head()[col][0]))
X.head()

# %%
generator.discrete_features

# %%
relevance_vector.sort_values(ascending=False)

# %%
generator.clusters

# %%
selector.interactions

# %%
selector.get_ranking()

# %%
selector.score_map

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
