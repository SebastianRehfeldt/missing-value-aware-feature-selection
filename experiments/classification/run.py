# %%
import os
import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import f1_score

from project import EXPERIMENTS_PATH
from project.utils import DataLoader, Data
from project.utils import introduce_missing_values, scale_data
from experiments.classification.utils import get_selectors, get_classifiers

# LOAD DATA AND DEFINE SELECTORS AND CLASSIFIERS
name = "ionosphere"
FOLDER = os.path.join(EXPERIMENTS_PATH, "classification", "incomplete", name)
os.makedirs(FOLDER, exist_ok=True)

data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
data = scale_data(data)
data.shuffle_rows(seed=42)

names = ["rar", "mrmr"]
classifiers = ["knn", "tree"]

k_s = [i + 1 for i in range(10)]
missing_rates = [0.2 * i for i in range(5)]

times = {mr: defaultdict(list) for mr in missing_rates}
for mr in missing_rates:
    data_copy = deepcopy(data)
    data_copy = introduce_missing_values(data_copy, missing_rate=mr, seed=42)

    scores_clf = {k: defaultdict(list) for k in k_s}
    scores = {algo: deepcopy(scores_clf) for algo in classifiers}
    splits = data_copy.split()
    for train, test in deepcopy(splits):

        selectors = get_selectors(train, names, max(k_s))
        for i_s, selector in enumerate(selectors):
            start = time()
            selector.fit(train.X, train.y)
            t = time() - start
            times[mr][names[i_s]].append(t)
            print("fitted", t)

            for k in k_s:
                X_train = selector.transform(train.X, k)
                X_test = selector.transform(test.X, k)

                f_types = train.f_types[X_train.columns]
                transformed_data = Data(X_train, train.y, f_types,
                                        train.l_type, X_train.shape)

                clfs = get_classifiers(transformed_data, classifiers)
                for i_c, clf in enumerate(clfs):
                    clf.fit(X_train, train.y)
                    y_pred = clf.predict(X_test)
                    f1 = f1_score(test.y, y_pred, average="micro")
                    scores[classifiers[i_c]][k][names[i_s]].append(f1)
                    print(f1)

    for clf in classifiers:
        means = pd.DataFrame(scores[clf]).applymap(np.mean).T
        stds = pd.DataFrame(scores[clf]).applymap(np.std).T
        means.to_csv(
            os.path.join(FOLDER, "mean_{:s}_{:.2f}.csv".format(clf, mr)))
        stds.to_csv(
            os.path.join(FOLDER, "std_{:s}_{:.2f}.csv".format(clf, mr)))

mean_times = pd.DataFrame(times).applymap(np.mean).T
mean_times.to_csv(os.path.join(FOLDER, "mean_times.csv"))
std_times = pd.DataFrame(times).applymap(np.std).T
std_times.to_csv(os.path.join(FOLDER, "std_times.csv"))
