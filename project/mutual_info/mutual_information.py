"""
    Utils class for estimating mutual information using PDS
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
from project import Data
from project.randomKNN.knn import KNN
from project.utils.assertions import assert_data


def _get_mi_cc(data):
    """
    Estimate mutual information for continous label types and at least one continous feature
    Checks how many samples are inside a given radius

    Arguments:
        data {data} -- Data Object for estimation
    """
    k = 6
    nx = np.ones(data.shape[0]) * k
    ny = np.ones(data.shape[0]) * k

    D_x = KNN.get_dist_matrix(data.X, data.f_types, data.X)
    inversed_data = data.inverse()
    D_y = KNN.get_dist_matrix(
        inversed_data.X, inversed_data.f_types, inversed_data.X)
    D_x.sort()
    D_y.sort()

    for row in range(data.shape[0]):
        # Get distances inside features and labels
        dist_x = D_x[row, :]
        dist_y = D_y[row, :]

        # Update statistics if sample contains non-nan values
        radius = max(dist_x[k+1], dist_y[k+1])
        if not np.isinf(radius):
            nx[row] = (dist_x <= radius).sum() - 1
            ny[row] = (dist_y <= radius).sum() - 1

    mi = digamma(data.shape[0]) + digamma(k) - 1/k - \
        digamma(np.mean(nx)) - digamma(np.mean(ny))
    return max(mi, 0)


def _get_mi_cd(data):
    """
    Estimate mutual information for discrete label types and at least one continous feature
    Checks how many samples of the same type are inside a given radius

    Arguments:
        data {data} -- Data Object for estimation
    """
    k = 6
    n = np.ones(data.shape[0]) * k
    m = np.ones(data.shape[0]) * k

    D = KNN.get_dist_matrix(data.X, data.f_types, data.X)
    D.sort()

    r_cache = {}
    n_cache = {}
    for row in range(data.shape[0]):
        # Get radius for k nearest neighbors within same class
        label = data.y[row]
        radius = r_cache.get(label)
        if radius is None:
            dist_cond = D[row, data.y == label]
            max_k = min(k+1, len(dist_cond) - 1)
            radius = dist_cond[max_k]
            r_cache[label] = radius
            n_cache[label] = len(dist_cond)

        # Get distances for all samples
        dist_full = D[row, :]

        # Update statistics if sample contains non-nan values
        if not np.isinf(radius):
            m[row] = (dist_full <= radius).sum() - 1
            n[row] = n_cache.get(label)

    mi = digamma(data.shape[0]) - np.mean(digamma(n)) + \
        digamma(k) - np.mean(digamma(m))
    return max(mi, 0)


def _get_mi_dd(data):
    """
    Estimate MI between discrete feature and discrete label
    Only works for 1-d features and creates class for missing values

    Arguments:
        data {data} -- Data object used for estimation
    """
    mi = mutual_info_score(data.X.iloc[:, 0], data.y)
    return max(mi, 0)


def get_mutual_information(data):
    """
    Estimate mutual information using different approaches

    Arguments:
        data {data} -- Data object for estimation
    """
    data = assert_data(data)

    # Estimate MI for numerical features only
    if not "nominal" in data.f_types.values:
        if data.l_type == "nominal":
            return _get_mi_cd(data)
        else:
            return _get_mi_cc(data)

    # Estimate MI when there are nominal features as well
    else:
        # Estimate MI for nominal targets
        if data.l_type == "nominal":
            # When numerical features are present, distances can be calculated
            if "numeric" in data.f_types.values:
                return _get_mi_cd(data)

            # Estimate MI for each nominal feature and return mean
            mi_s = np.zeros(data.shape[1])
            for i, col in enumerate(data.X):
                selected_data = data.select(col)
                mi_s[i] = _get_mi_dd(selected_data)
            return np.mean(mi_s)

        # Estimate MI for numerical targets
        else:
            # When numerical features are present, distances can be calculated
            if "numeric" in data.f_types.values:
                return _get_mi_cc(data)
            else:
                # Estimate MI for each nominal feature and return mean
                # Features and labels are flipped to match get_mi_cd
                # TODO check symmilarity assumption for mi estimation
                mi_s = np.zeros(data.shape[1])
                for i, col in enumerate(data.X):
                    inversed_data = data.select_inverse(col)
                    mi_s[i] = _get_mi_cd(inversed_data)
                return np.mean(mi_s)

    print(data)
    sys.exit("MI messed up types")
    return -1
