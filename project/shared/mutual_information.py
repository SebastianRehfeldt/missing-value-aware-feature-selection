# %%
import sys
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader
from project.utils.data_modifier import introduce_missing_values
from project.utils.data_scaler import scale_data
from project.shared.neighbors import Neighbors

data_loader = DataLoader()
data = data_loader.load_data("analcatdata_reviewer", "arff")
data = data_loader.load_data("iris", "arff")
data = scale_data(data)
data = introduce_missing_values(data)
data.features.head()

# %%
from scipy.special import digamma
from project import Data


def _get_mi_cc(data):
    nn_x = Neighbors(data)

    new_data = data._replace(
        features=data.labels.to_frame(), f_types=pd.Series(data.l_type))
    nn_y = Neighbors(new_data)

    k = 6
    nx = np.ones(data.shape[0]) * k
    ny = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        sample_x = data.features.iloc[i]
        dist_x = nn_x.partial_distances(sample_x)
        dist_x.sort()

        sample_y = new_data.features.iloc[i]
        dist_y = nn_y.partial_distances(sample_y)
        dist_y.sort()

        max_dist = max(dist_x[k+1], dist_y[k+1])
        if not np.isinf(max_dist):
            nx[i] = (dist_x <= max_dist).sum() - 1
            ny[i] = (dist_y <= max_dist).sum() - 1

    mi = digamma(data.shape[0]) + digamma(k) - 1/k - \
        digamma(np.mean(nx)) - digamma(np.mean(ny))
    return max(mi, 0)


def _get_mi_cd(data):
    nn = Neighbors(data)
    k = 6
    n = np.ones(data.shape[0]) * k
    m = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        # We are interested in nearest neighbors within same class
        label = data.labels[i]
        features = data.features[data.labels == label].reset_index(drop=True)
        new_data = data._replace(features=features, shape=features.shape)
        nn_cond = Neighbors(new_data)

        sample = data.features.iloc[i]
        dist_cond = nn_cond.partial_distances(sample)
        dist_cond.sort()
        max_k = min(k+1, len(dist_cond) - 1)
        max_dist = dist_cond[max_k]

        dist_full = nn.partial_distances(sample)

        if not np.isinf(max_dist):
            m[i] = (dist_full <= max_dist).sum() - 1
            n[i] = new_data.features.shape[0]

    mi = digamma(data.shape[0]) - np.mean(digamma(n)) + \
        digamma(k) - np.mean(digamma(m))
    return max(mi, 0)


def _get_mi_dd(data):
    # Works for 1d only
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(data.features.iloc[:, 0], data.labels)
    return max(mi, 0)


def get_mutual_information(data):
    # Gets a dataframe as input which has >= 1 columns
    if not "nominal" in data.f_types.values:
        if data.l_type == "nominal":
            return _get_mi_cd(data)
        else:
            return _get_mi_cc(data)
    else:
        if data.l_type == "nominal":
            if "numeric" in data.f_types.values:
                return _get_mi_cd(data)

            mi_s = np.zeros(data.shape[1])
            for i, col in enumerate(data.features):
                features = data.features[col].to_frame()
                types = pd.Series(data.f_types[col])
                selected_data = data._replace(
                    features=features, f_types=types, shape=features.shape)
                mi_s[i] = _get_mi_dd(selected_data)
            return np.mean(mi_s)
        else:
            if "numeric" in data.f_types.values:
                return _get_mi_cc(data)
            else:
                mi_s = np.zeros(data.shape[1])
                for i, col in enumerate(data.features):
                    new_X = data.labels.to_frame()
                    new_X.columns = [col]
                    new_y = data.features[col]
                    f_types = pd.Series(data.l_type, [col])
                    l_type = data.f_types[col]

                    inversed_data = Data(
                        new_X, new_y, f_types, l_type, new_X.shape)
                    mi_s[i] = _get_mi_cd(inversed_data)
                return np.mean(mi_s)

    print(data)
    sys.exit("MI messed up types")
    return -1


def get_mis(data):
    mi_s = np.zeros(data.shape[1])
    for i, col in enumerate(data.features):
        features = data.features[col].to_frame()
        types = pd.Series(data.f_types[col], [col])

        col = ["Film", "Jeffrey_Lyons"]
        col = ["petallength", "petalwidth"]
        features = data.features[col]
        types = pd.Series(data.f_types[col])

        selected_data = data._replace(
            features=features, f_types=types, shape=features.shape)

        mi_s[i] = get_mutual_information(selected_data)
    return mi_s


get_mis(data)
