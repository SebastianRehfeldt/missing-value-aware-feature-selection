# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
name = "madelon"
name = "semeion"
name = "ionosphere"
name = "analcatdata_reviewer"
name = "boston"
name = "credit-approval"
name = "iris"
data = data_loader.load_data(name, "arff")
data.shape

# %%
from project.utils.data_modifier import introduce_missing_values
from project.utils.data_scaler import scale_data

data = introduce_missing_values(data, missing_rate=0.25)
data = scale_data(data)

# %%
from time import time
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from project.randomKNN.random_knn import RKNN
from project.randomKNN.knn import KNN
from project.tree.tree import Tree
from project.utils.imputer import Imputer
from project.mutual_info.mi_filter import MI_Filter
from project.shared.sfs import SFS

rknn = RKNN(data.f_types, data.l_type, data.shape)
mi = MI_Filter(data.f_types, data.l_type, data.shape)
sfs = SFS(data.f_types, data.l_type, data.shape, method="mi")
knn = KNN(data.f_types, data.l_type)
tree = Tree(data.to_table().domain)
imputer = Imputer(data.f_types, method="mice")

pipe1 = Pipeline(steps=[('reduce', rknn), ('classify', knn)])
pipe2 = Pipeline(steps=[
    ("imputer", imputer),
    ('classify', knn),
])
pipe3 = Pipeline(steps=[('classify', knn)])
pipe4 = Pipeline(steps=[('reduce', mi), ('classify', knn)])
pipe5 = Pipeline(steps=[('reduce', sfs), ('classify', knn)])
pipe6 = Pipeline(steps=[('classify', tree)])
pipe7 = Pipeline(steps=[
    ("imputer", imputer),
    ('reduce', rknn),
    ('classify', knn),
])
"""
X_new = rknn.fit_transform(data.X, data.y)
types = pd.Series(data.f_types, X_new.columns.values)
new_data = data.replace(True, X=X_new, shape=X_new.shape, f_types=types)
new_knn = KNN(new_data.f_types, new_data.l_type)

pipe8 = Pipeline(steps=[
    ("imputer", Imputer(new_data.f_types, method="mice")),
    ('classify', new_knn),
])

pipelines = [pipe1, pipe2, pipe3, pipe4, pipe5, pipe6, pipe7, pipe8]
"""
pipelines = [pipe4, pipe5]

scores = []
times = []
cv = StratifiedKFold(data.y, n_folds=4, shuffle=True)
scoring = "accuracy" if data.l_type == "nominal" else "neg_mean_squared_error"

for pipe in pipelines:
    start = time()
    score = cross_val_score(
        pipe, data.X, data.y, cv=cv, scoring=scoring, n_jobs=1)
    scores.append(score)
    times.append(time() - start)

print("Results\n\n")
for i, score in enumerate(scores):
    print("Pipe{:d} with mean {:.3f} took {:.3f}s.".format(
        i + 1, np.mean(score), times[i]))
    print("Detailed scores: ")
    print(score)
    print("\n")
