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

mr = 0
data = introduce_missing_values(data, missing_rate=mr)
data = scale_data(data)

gold_ranking = [
    ('a05', 0.43040407748046167), ('a06', 0.41160532164225955),
    ('a29', 0.3919975515222594), ('a33', 0.39007589467758935),
    ('a03', 0.38533617145294263), ('a08',
                                   0.37898307236412015), ('a21',
                                                          0.37265243615230303),
    ('a14', 0.3659502837628848), ('a34',
                                  0.36153979486970356), ('a31',
                                                         0.35639223764674177),
    ('a07', 0.3563310760238173), ('a16',
                                  0.3494314037169902), ('a13',
                                                        0.3459749559004396),
    ('a27', 0.34407479403668073), ('a24',
                                   0.34082422076282215), ('a10',
                                                          0.3325634521842949),
    ('a23', 0.32934318358727), ('a15',
                                0.3265554952863316), ('a18',
                                                      0.32593687128161913),
    ('a28', 0.32064778848053194), ('a09',
                                   0.315575654727455), ('a22',
                                                        0.31395077481300265),
    ('a04', 0.31277623774633556), ('a25',
                                   0.3119327566593308), ('a26',
                                                         0.3077594407394208),
    ('a17', 0.29558848944261495), ('a32',
                                   0.2938056057492842), ('a11',
                                                         0.28444532373011816),
    ('a12', 0.28333549693846855), ('a30',
                                   0.2826382540608627), ('a20',
                                                         0.2616264813941282),
    ('a19', 0.25366966449009665), ('a01',
                                   0.2513849409434004), ('a02',
                                                         0.10778976963332028)
]

zlst = list(zip(*gold_ranking))
gold_ranking = pd.Series(zlst[1], index=zlst[0])

# %%
from project.rar.rar import RaR
from experiments.metrics import calc_ndcg
from project.utils.data import DataGenerator
from copy import deepcopy
from project.utils.imputer import Imputer

n_runs = 1
seeds1 = [0]
seeds1 = [42, 0, 113, 98, 234, 143, 1, 20432, 4357, 12]
seeds = [3] * 10
missing_rates = [0.5, 0.7, 0.8, 0.9]
missing_rates = [0]
missing_rates = [0.05 * i for i in range(20)]
missing_rates = [0.1 * i for i in range(10)]
avgs = np.zeros(len(missing_rates))
stds = np.zeros(len(missing_rates))
sums = np.zeros(len(missing_rates))
data_orig = deepcopy(data)

is_synthetic = True
generator = DataGenerator(n_samples=3000, n_relevant=1)

for j, mr in enumerate(missing_rates):
    print("======== {:.2f} ========".format(mr))
    ndcgs = np.zeros(n_runs)
    for i in range(n_runs):
        if is_synthetic:
            generator.set_seed(seeds[i])
            data_orig, relevance_vector = generator.create_dataset()
            imputer = Imputer(data_orig.f_types, strategy="knn")

        data_copy = deepcopy(data_orig)
        data_copy = introduce_missing_values(data_copy, mr, seed=seeds1[i])
        # data_copy = imputer.complete(data_copy)

        start = time()
        rar = RaR(
            data_copy.f_types,
            data_copy.l_type,
            data_copy.shape,
            n_jobs=1,
            approach="fuzzy",
            n_targets=0,
            n_subspaces=0,
            subspace_size=(1, 3),
            contrast_iterations=250,
            alpha=min(0.1, 0.02 * (1 / (1 - mr))),
            redundancy_approach="arvind",
            weight=0.1 * mr,
            random_state=seeds1[i],
            cache_enabled=False,
            min_samples=5,
            min_slices=30,
            resamples=10,
        )

        rar.fit(data_copy.X, data_copy.y)
        # pprint(rar.get_ranking())
        # print(time() - start)
        ranking = [k for k, v in rar.get_ranking() if v > 1e-4]
        ndcgs[i] = calc_ndcg(relevance_vector, ranking, False)
        # print(ndcgs[i])

    avgs[j] = np.mean(ndcgs)
    stds[j] = np.std(ndcgs)
    sums[j] = np.sum([v for k, v in rar.get_ranking()])

    print(rar.hics.evaluate_subspace(["f2"]))
    print(rar.hics.evaluate_subspace(["f0"]))
    print(rar.hics.evaluate_subspace(["f13", "f17"]))
    print(rar.hics.evaluate_subspace(["f11", "f3"]))

rar_results = pd.DataFrame(avgs, columns=["AVG"], index=missing_rates)
rar_results["STD"] = stds
rar_results["SUM"] = sums
rar_results = rar_results.T
rar_fuz = rar_results.copy()
# rar_results

# %%
print(rar.hics.evaluate_subspace(["f2"]))
print(rar.hics.evaluate_subspace(["f0"]))
print(rar.hics.evaluate_subspace(["f13", "f17"]))
print(rar.hics.evaluate_subspace(["f11", "f3"]))

# %%
relevance_vector.sort_values(ascending=False)

# %%
generator.get_clusters()

# %%
rar.get_ranking()

# %%
rar.score_map

# %%
i = 0

print(rar.hics.evaluate_subspace(["f11", "f3"]))
print(
    np.unique(
        rar.hics.get_cached_slices(["f11", "f3"])[0][i, :],
        return_counts=True))
print(np.sum(rar.hics.get_cached_slices(["f11", "f3"])[0][i, :]))

# %%
rar.hics.alphas_d
rar.hics.n_select_d

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
