# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")
data = data_loader.load_data("boston", "arff")
data = data_loader.load_data("credit-approval", "arff")

"""
print(data.y.head())
print(data.f_types.head())
print(data.l_type)
"""
print(data.X.head())


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
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from project.randomKNN.random_knn import RKNN
from project.randomKNN.knn import KNN
from project.tree.tree import Tree
from project.utils.imputer import Imputer
from project.mutual_info.mi_filter import MI_Filter

knn = KNN(data.f_types, data.l_type)

pipe1 = Pipeline(steps=[
    ('reduce', RKNN(data)),
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
    ('reduce', MI_Filter(data)),
    ('classify', knn)
])

pipe5 = Pipeline(steps=[
    ('classify', Tree(data))
])

pipelines = [pipe1, pipe2, pipe3, pipe4, pipe5]
pipelines = [pipe5]

scores = []
cv = StratifiedKFold(data.y, n_folds=2, shuffle=True)
scoring = "accuracy" if data.l_type == "nominal" else "mean_squared_error"
for pipe in pipelines:
    scores.append(cross_val_score(pipe, data.X, data.y,
                                  cv=cv, scoring=scoring, n_jobs=1))

for score in scores:
    print(np.mean(score), score)
