# %%
from project.utils.data import DataGenerator
from project.utils.plots import plot_nan_correlation
from experiments.metrics import calc_ndcg

gen = DataGenerator(None, n_informative_missing=2, missing_rate=0)
data, rel = gen.create_dataset()
"""
print(gen.clusters)
print(gen.informative_missing)
print(rel.sort_values(ascending=False))
"""

# %%
from copy import deepcopy
from project.utils.data_modifier import introduce_missing_values

data2 = deepcopy(data)
data2 = introduce_missing_values(data, 0.5, "correlated")

# %%
plot_nan_correlation(data2)

# %%
from project.rar.rar import RaR

rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    approach="deletion",
    redundancy_approach="tom",
    active_sampling=True,
    boost=0.1,
)
rar.fit(data.X, data.y)
ranking = rar.get_ranking()
ranking = [k for k, v in dict(ranking).items() if v > 0]
calc_ndcg(rel, ranking)

# %%
rel.sort_values(ascending=False)

# %%
rar.get_ranking()

# %%
rar.interactions
rar.get_ranking()
rar.nan_correlation.style.background_gradient("coolwarm")
rar.nan_correlation.loc[["f4", "f2"], ["f18", "f7", "f10"]].max()

# %%
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection import Filter, RKNN

selector = [
    Ranking(data.f_types, data.l_type, data.shape, eval_method="myrelief"),
    Ranking(data.f_types, data.l_type, data.shape, eval_method="cfs"),
    Ranking(data.f_types, data.l_type, data.shape, eval_method="mrmr"),
    Ranking(data.f_types, data.l_type, data.shape, eval_method="fcbf"),
    Embedded(data.f_types, data.l_type, data.shape),
    Filter(data.f_types, data.l_type, data.shape),
    RKNN(data.f_types, data.l_type, data.shape),
]

for s in selector:
    d = deepcopy(data)
    s.fit(d.X, d.y)
    ranking = s.get_ranking()
    print(ranking)
    ranking = [k for k, v in dict(ranking).items() if v > 0]
    print(calc_ndcg(rel, ranking))
