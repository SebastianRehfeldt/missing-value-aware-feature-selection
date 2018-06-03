# %%
from time import time
from pprint import pprint
from project.utils import DataLoader

if __name__ == '__main__':

    data_loader = DataLoader()
    name = "analcatdata_reviewer"
    name = "credit-approval"
    name = "madelon"
    name = "musk"
    name = "semeion"
    name = "boston"
    name = "ionosphere"
    name = "iris"
    name = "isolet"
    data = data_loader.load_data(name, "arff")
    print(data.shape, flush=True)

    # %%
    from project.utils import introduce_missing_values, scale_data

    data = introduce_missing_values(data, missing_rate=0)
    data = scale_data(data)

    # %%
    from project.rar import RaR

    start = time()
    rar = RaR(
        data.f_types,
        data.l_type,
        data.shape,
        n_jobs=1,
        contrast_iterations=10000,
    )
    rar.fit(data.X, data.y)
    print(time() - start)
    #pprint(rar.feature_importances)
