# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.load_data("analcatdata_reviewer", "arff")


# %%
from project.utils.data_modifier import introduce_missing_values

data = introduce_missing_values(data)


# %%
from project.utils.data_scaler import scale_data

data = scale_data(data)
data.features.head()


# %%
from project.shared.neighbors import Neighbors


def _get_mi_cc(data):
    from scipy.special import digamma

    nn_x = Neighbors(data)
    new_types = data.labels
    new_types[0], new_types[len(new_types)-1] = new_types[len(
        new_types)-1], new_types[0]
    nn_y = Neighbors(data._replace(features=data.labels,
                                   types=new_types))
    k = 6
    nx = np.ones(data.shape[0]) * k
    ny = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        sample_x = data.features.iloc[i]
        dist_x = nn_x.partial_distances(sample_x, True)
        dist_x.sort()

        sample_y = data.labels[i]
        dist_y = nn_y.partial_distances(sample_y, True)
        dist_y.sort()

        if not np.isnan(sample_x):
            epsilon = max(dist_x[k+1], dist_y[k+1])

            nx[i] = (dist_x <= epsilon).sum() - 1
            ny[i] = (dist_y <= epsilon).sum() - 1

    mi = digamma(data.shape[0]) + digamma(k) - 1/k - \
        digamma(np.mean(nx)) - digamma(np.mean(ny))
    return max(mi, 0)


def _get_mi_cd(data):
    from scipy.special import digamma
    from project import Data

    nn = Neighbors(data)
    k = 6
    n = np.ones(data.shape[0]) * k
    m = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        label = data.labels[i]
        new_features = data.features[data.labels ==
                                     label].reset_index(drop=True)
        new_data = Data(new_features, data.labels,
                        data.types, new_features.shape)
        nn_cond = Neighbors(new_data)

        sample = data.features.iloc[i]
        dist_cond = nn_cond.partial_distances(sample, True)
        dist_cond.sort()
        max_dist = dist_cond[k + 1]

        dist_full = nn.partial_distances(sample, True)
        if not np.isnan(sample):
            m[i] = (dist_full <= max_dist).sum() - 1
            n[i] = new_data.features.shape[0]

    mi = digamma(data.shape[0]) - np.mean(digamma(n)) + \
        digamma(k) - np.mean(digamma(m))
    return mi


def _get_mi_dd(data):
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(data.features, data.labels)
    return mi


def get_mi(data):
    mi_s = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        selected_data = data._replace(features=data.features.iloc[:, i])
        mi_s[i] = _get_mi_dd(selected_data)
    return mi_s


get_mi(data)
