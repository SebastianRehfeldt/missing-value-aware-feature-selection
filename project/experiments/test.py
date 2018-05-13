# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")

"""
print(data.X.head())
print(data.y.head())
print(data.f_types.head())
print(data.l_type)
"""


# %%
from project.utils.data_modifier import introduce_missing_values

data = introduce_missing_values(data)
data.X.head()


# %%
from project.utils.data_scaler import scale_data

data = scale_data(data)
data.X.head()

# %%
"""
from sklearn.preprocessing import LabelEncoder
y = pd.Series(LabelEncoder().fit_transform(data.y), name=data.y.name)
data = data.replace(y=y, l_type="numeric")
data.y.head()
"""


# %%
"""
from project.utils.imputer import Imputer

imputer = Imputer(data)
data_complete = imputer.complete()
data_complete.X.head()
"""


# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from project.randomKNN.random_knn import RKNN
from project.randomKNN.knn import KNN
from project.tree.tree import Tree
from project.utils.imputer import Imputer
from project.mutual_info.mi_filter import MI_Filter

rknn = RKNN(data, method="classifier")
mi = MI_Filter(data)
knn = KNN(data.f_types, data.l_type)
tree = Tree(data.f_types, data.l_type)
y = LabelEncoder().fit_transform(data.y)
y = [str(l) for l in y]
y = pd.Series(y)
cv = StratifiedKFold(y, n_folds=4, shuffle=True)

pipe1 = Pipeline(steps=[
    ('reduce', rknn),
    ('classify', knn)
])

pipe2 = Pipeline(steps=[
    ("imputer", Imputer(data)),
    ('classify', knn),
])

pipe3 = Pipeline(steps=[
    ('classify', knn)
])

pipe4 = Pipeline(steps=[
    ('reduce', mi),
    ('classify', knn)
])

pipe5 = Pipeline(steps=[
    ('classify', tree)
])

pipelines = [pipe1, pipe2, pipe3, pipe4]
pipelines = [pipe3, pipe5]

scores = []
for pipe in pipelines:
    scores.append(cross_val_score(pipe, data.X, y,
                                  cv=cv, scoring="accuracy", n_jobs=-1))

for score in scores:
    print(np.mean(score), score)
