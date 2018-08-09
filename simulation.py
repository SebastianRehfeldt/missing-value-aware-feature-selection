# %%
from project.utils.data import DataGenerator
from project.utils.plots import plot_nan_correlation, plot_nan_percentage

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
plot_nan_percentage(data2)

# %%
from project.rar.rar import RaR

rar = RaR(
    data.f_types,
    data.l_type,
    data.shape,
    approach="deletion",
    redundancy_approach="arvind",
    active_sampling=True,
    boost=0.1,
)
rar.fit(data.X, data.y)
rar.get_ranking()

# %%
rel.sort_values(ascending=False)

# %%
print(gen.clusters, gen.informative_missing)

# %%
rar.interactions
rar.get_ranking()

# %%
rar.hics.evaluate_subspace(["f16"])

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
    s.fit(data.X, data.y)
    print(s.get_ranking())