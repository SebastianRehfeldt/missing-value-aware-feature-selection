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
name = "ionosphere"
data = data_loader.load_data(name, "arff")
data = scale_data(data)
print(data.shape, flush=True)

#imputer = Imputer(data.f_types, strategy="mice")
#d = imputer.complete(data)

# %%
data = introduce_missing_values(data, 0.8)
d = deepcopy(data)
rar = RaR(
    d.f_types,
    d.l_type,
    d.shape,
    approach="fuzzy",
    weight_approach="new",
    boost=0.1,
    nullity_corr_boost=0.1,
    active_sampling=True,
    n_subspaces=50,
)
rar.fit(d.X, d.y)
ranking = [k for k, v in rar.get_ranking() if v > 1e-4]
#print(calc_ndcg(relevance_vector, ranking, True))
print(rar.hics.evaluate_subspace(["a05"]))
print(rar.hics.evaluate_subspace(["a05", "a06"]))
pprint(rar.get_ranking())

# %%
rar.get_params()

# %%
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
relevance_vector = pd.Series(zlst[1], index=zlst[0])

# %%
n_runs = 3
seeds = [42, 0, 113, 98, 234, 143, 1, 20432, 4357, 12]
seeds = [0] * n_runs
missing_rates = [0.1 * i for i in range(10)]
missing_rates = [0.2]
missing_rates = [0.2 * i for i in range(1, 5)]
avgs = np.zeros(len(missing_rates))
stds = np.zeros(len(missing_rates))
sums = np.zeros(len(missing_rates))
data_orig = deepcopy(data)

is_synthetic = True
generator = DataGenerator(n_samples=500)

for j, mr in enumerate(missing_rates):
    print("======== {:.2f} ========".format(mr))
    ndcgs = np.zeros(n_runs)
    for i in range(n_runs):
        if is_synthetic:
            generator.set_seed(seeds[i])
            data_orig, relevance_vector = generator.create_dataset()
            imputer = Imputer(data_orig.f_types, strategy="knn")

        data_copy = deepcopy(data_orig)
        data_copy = introduce_missing_values(data_copy, mr, seed=seeds[i])
        #data_copy = imputer.complete(data_copy)

        start = time()
        rar = RaR(
            data_copy.f_types,
            data_copy.l_type,
            data_copy.shape,
            approach="deletion",
            n_targets=0,
            #weight_approach="alpha",
            #random_state=seeds[j],
            imputation_method="mice",
            boost=0,
            active_sampling=True,
            min_samples=5,
        )

        rar.fit(data_copy.X, data_copy.y)
        # pprint(rar.get_ranking())
        # print(time() - start)
        ranking = [k for k, v in rar.get_ranking() if v > 1e-4]
        ndcgs[i] = calc_ndcg(relevance_vector, ranking, False)
        print(ndcgs[i])

    avgs[j] = np.mean(ndcgs)
    stds[j] = np.std(ndcgs)
    sums[j] = np.sum([v for k, v in rar.get_ranking()])
    print(avgs[j], stds[j])

rar_results = pd.DataFrame(avgs, columns=["AVG"], index=missing_rates)
rar_results["STD"] = stds
rar_results["SUM"] = sums
rar_results = rar_results.T
rar_results.T
# %%
rar.hics.evaluate_subspace(["f1"])
relevance_vector.sort_values(ascending=False)

# %%
rar.get_ranking()
rar.interactions
print(rar.hics.evaluate_subspace(["f7", "f8"]))
print(rar.hics.evaluate_subspace(["f7"]))

# %%
rar.get_ranking()
# %%
generator.clusters
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
