# %%
from project.utils import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")

from project.utils import introduce_missing_values, scale_data

data = introduce_missing_values(data, missing_rate=0.25)
data = scale_data(data)
data.shape

# %%
from project.shared import evaluate_subspace
domain = data.to_table().domain

# %%
from pyswarm import pso


def objective(x, **kwargs):
    data = kwargs.get("data")
    domain = kwargs.get("domain")

    if (x <= 0.6).all():
        return 1

    subspace = data.X.columns[x > 0.6].tolist()
    X, types = data.get_subspace(subspace)
    score = evaluate_subspace(
        X, data.y, types, data.l_type, domain, method="tree")

    return 1 - score


lb = [0] * 4
ub = [1] * 4
params = {
    "data": data,
    "domain": data.to_table().domain,
    "params": {},
}

default_options = {
    "omega": 0.729844,
    "phip": 1.49618,
    "phig": 1.49618,
    "swarmsize": 50,
    "maxiter": 100,
    "debug": True,
}

default_options.update({
    "swarmsize": int(4**2 / 2),
    "maxiter": 4**2,
})

x_opt, f_opt = pso(objective, lb, ub, **default_options, kwargs=params)

subspace = data.X.columns[x_opt > 0.6].tolist()
subspace