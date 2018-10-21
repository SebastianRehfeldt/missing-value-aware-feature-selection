# %%
# LOAD DATASET FROM OPENML OR UCI
import numpy as np
from time import time
from project.rar.rar import RaR
from project.utils import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("ionosphere", "arff")
print("Dataset dimensions:", data.shape)
data.X.head()

# %%
# STANDARDIZE DATA AND INTRODUCE MISSING VALUES
from project.utils import introduce_missing_values, scale_data

data = scale_data(data)
data = introduce_missing_values(data, missing_rate=0)
data.X.head()

# %%
# CREATE AND PRINT RANKING USING RAR
t = time()
rar = RaR(data.f_types, data.l_type, data.shape)
rar.fit(data.X, data.y)
print("Time for ranking:", time() - t)
rar.get_ranking()

# %%
# GENERATE SYNTHETIC DATA
from project.utils.data import DataGenerator

generator = DataGenerator(
    n_samples=500, n_relevant=3, n_clusters=1, n_discrete=10)
data_syn, relevance_vector = generator.create_dataset()
relevance_vector.sort_values(ascending=False)

# %%
# FIT RAR ON SYNTHETIC DATA
t = time()
rar = RaR(data_syn.f_types, data_syn.l_type, data_syn.shape)
rar.fit(data_syn.X, data_syn.y)
print("Time for ranking:", time() - t)
rar.get_ranking()

# %%
from copy import deepcopy
from experiments.metrics import calc_cg
from project.utils.imputer import Imputer
from project.feature_selection.ranking import Ranking
from project.feature_selection import Filter, RKNN, SFS
from project.feature_selection.embedded import Embedded
from project.feature_selection.orange import Orange
from project.feature_selection.baseline import Baseline

n_runs = 1
seeds = [42, 0, 113, 98, 234, 143, 1, 20432, 4357, 12]
missing_rates = [0.1 * i for i in range(0, 10)]

use_rar = True
shuffle_seed = 0
add_noise = False
is_synthetic = True
should_impute = False

for j, mr in enumerate(missing_rates):
    print("======== {:.2f} ========".format(mr))
    for i in range(n_runs):
        # Data Loading
        if is_synthetic:
            generator.set_seed(seeds[i])
            data_orig, relevance_vector = generator.create_dataset()
        else:
            data_orig = deepcopy(data)
            if add_noise:
                data_orig = data._add_noisy_features(seed=shuffle_seed + 1)

        # Missing Value Simulation
        data_copy = deepcopy(data_orig)
        data_copy = introduce_missing_values(data_copy, mr, seed=seeds[i])

        # Imputation
        if should_impute:
            imputer = Imputer(data_orig.f_types, strategy="soft")
            data_copy = imputer.complete(data_copy)

        # Shuffle columns
        data_copy.shuffle_columns(seed=shuffle_seed)
        shuffle_seed += 1

        # Fit Feature Selector
        start = time()
        selector = RaR(data_copy.f_types, data_copy.l_type, data_copy.shape)

        if not use_rar:
            selector = Ranking(data_copy.f_types, data_copy.l_type,
                               data_copy.shape)

        selector.fit(data_copy.X, data_copy.y)
        print("Time", time() - start)
        ranking = [k for k, v in selector.get_ranking()]

        if is_synthetic:
            n_relevant = np.count_nonzero(relevance_vector.values)
            print("CG:", calc_cg(relevance_vector, ranking)[n_relevant])
        else:
            print(selector.get_ranking())

# %%
# FIT ON REAL-WORLD DATA AND PLOT CORRELATION
import pandas as pd

data_loader = DataLoader()
data = data_loader.load_data("ionosphere", "arff")

rar = RaR(data.f_types, data.l_type, data.shape)
X_new = rar.fit_transform(data.X, data.y, k=4)
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data.replace(True, X=X_new, shape=X_new.shape, f_types=types)
X_new.corr().style.background_gradient()

# %%
# EVALUATE CLASSIFICATION
from project.classifier import KNN
from sklearn.metrics import make_scorer, f1_score
from sklearn.cross_validation import StratifiedKFold, cross_val_score

knn = KNN(new_data.f_types, new_data.l_type, knn_neighbors=6)

cv = StratifiedKFold(new_data.y, n_folds=5, shuffle=True)
scorer = make_scorer(f1_score, average="micro")

scores = cross_val_score(
    knn, new_data.X, new_data.y, cv=cv, scoring=scorer, n_jobs=3)
print(np.mean(scores), scores)
