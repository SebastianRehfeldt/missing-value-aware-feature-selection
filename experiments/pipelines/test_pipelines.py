# %%
import os
import numpy as np
import pandas as pd
from time import time

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, StratifiedKFold

from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.rar.rar import RaR
from project.classifier import KNN

from project import EXPERIMENTS_PATH
from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data

# TODO: test multiple missing rates, different k params
mr, k = 0, 2
name = "iris"
data_loader = DataLoader(ignored_attributes=["molecule_name"])
data = data_loader.load_data(name, "arff")
data = introduce_missing_values(data, missing_rate=mr)
data = scale_data(data)
data.shuffle_rows()

d = [data.f_types, data.l_type, data.shape]
rar = RaR(*d, k=k)
rknn = RKNN(*d, k=k)
mi = Filter(*d, k=k)
sfs = SFS(*d, k=k)
relief_sk = Ranking(*d, eval_method="myrelief")
fcbf_sk = Ranking(*d, eval_method="fcbf")
mrmr = Ranking(*d, eval_method="mrmr")
cfs = Ranking(*d, eval_method="cfs")
relief_o = Orange(*d, eval_method="relief")
fcbf_o = Orange(*d, eval_method="fcbf_o")
rf = Orange(*d, eval_method="rf")

knn = KNN(data.f_types, data.l_type, knn_neighbors=6)

pipelines = []
pipelines.append(Pipeline(steps=[('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', rar), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', rknn), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', mi), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', sfs), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', relief_sk), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', fcbf_sk), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', mrmr), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', cfs), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', relief_o), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', fcbf_o), ('classify', knn)]))
pipelines.append(Pipeline(steps=[('reduce', rf), ('classify', knn)]))

names = [
    "complete",
    "rar",
    "rknn",
    "mi",
    "sfs",
    "relief_sk",
    "fcbf_sk",
    "mrmr",
    "cfs",
    "relief_o",
    "fcbf_o",
    "rf",
]

scores = pd.DataFrame()
cv = StratifiedKFold(data.y, n_folds=3)

for i, pipe in enumerate(pipelines):
    start = time()
    res = cross_val_score(pipe, data.X, data.y, cv=cv, scoring="f1_micro")
    mean, std, t = np.mean(res), np.std(res), time() - start
    scores[names[i]] = pd.Series({"AVG": mean, "STD": std, "TIME": t})
    print("Results for:", names[i])
    print("Mean: {:.3f}\tStd: {:.3f}".format(mean, std))
    print("Detailed_scores:", res)
    print("Elapsed time:", t)
    print("\n\n_")
scores = scores.T
scores.sort_values(by="AVG", inplace=True, ascending=False)

FOLDER = os.path.join(EXPERIMENTS_PATH, "pipelines")
scores.to_csv(os.path.join(FOLDER, "{:s}.csv".format(name)))
