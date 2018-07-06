# %%
import pandas as pd

from project.utils import DataLoader
from project.utils import introduce_missing_values, scale_data
# from project.utils.imputer import Imputer

data_loader = DataLoader(ignored_attributes=["molecule_name"])
name = "madelon"
name = "boston"
name = "analcatdata_reviewer"
name = "credit-approval"  # standard config
name = "musk"  # standard config
name = "heart-c"  # 800 subspaces, alpha = 0,2, 100 iterations, (1,3)
name = "isolet"
name = "semeion"
name = "ionosphere"  # 800 subspaces, alpha=0.02, 250 iterations ,(1,3)
name = "iris"
data = data_loader.load_data(name, "arff")
print(data.shape, flush=True)

mr = 0
data = introduce_missing_values(data, missing_rate=mr)
data = scale_data(data)
data.shuffle_rows()

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from project.classifier import KNN
from project.rar.rar import RaR

knn = KNN(data.f_types, data.l_type, knn_neighbors=20)
rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    approach='partial',
    subspace_size=(1, 3))

pipe = Pipeline([('reduce_dim', rar), ('classify', knn)])

param_grid = [
    {
        'reduce_dim': [RaR(data.f_types, data.l_type, data.shape)],
        'reduce_dim__k': [2],
        'reduce_dim__contrast_iterations': [100, 250],
        'reduce_dim__alpha': [0.1, 0.02],
        'reduce_dim__n_iterations': [400, 800],
    },
]

scorer = make_scorer(f1_score, average="micro")
grid = GridSearchCV(
    pipe, cv=3, n_jobs=4, param_grid=param_grid, scoring=scorer)
grid.fit(data.X, data.y)

# %%
results = pd.DataFrame(grid.cv_results_).sort_values(
    by="mean_test_score", ascending=False)
results.to_csv("./experiments/grid/results.csv")

ranking = grid.best_estimator_.steps[0][1].get_ranking()
ranking = dict(ranking)
ranking = pd.DataFrame(
    data={"score": list(ranking.values())},
    index=ranking.keys(),
)
ranking.to_csv("./experiments/grid/ranking.csv")
