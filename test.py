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

mr = 0.1
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

ndcgs = np.zeros(5)
for i in range(5):
    start = time()
    rar = RaR(
        data.f_types,
        data.l_type,
        data.shape,
        n_jobs=1,
        approach="fuzzy",
        n_targets=0,
        n_subspaces=800,
        subspace_size=(1, 3),
        contrast_iterations=250,
        alpha=0.02,
        redundancy_approach="tom",
        weight=0.1,  #min(0.9, (1 - mr)**2),
        #random_state=42,
        cache_enabled=True,
        sample_slices=True,
        min_samples=3,
    )

    rar.fit(data.X, data.y)
    pprint(rar.get_ranking())
    # print(time() - start)
    ranking = [k for k, v in rar.get_ranking() if v > 1e-4]
    ndcgs[i] = calc_ndcg(gold_ranking, ranking)

print(np.mean(ndcgs), np.std(ndcgs))

# %%
np.histogram(rar.hics.deviations)

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
