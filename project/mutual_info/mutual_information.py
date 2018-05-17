"""
    Utils class for estimating mutual information using PDS
"""
import sys
import numpy as np
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
from project.utils import assert_df, assert_series, assert_types

from project.randomKNN.knn import KNN


def _get_mi_cc(X, y, f_types, l_type):
    """
    Estimate mutual information for continous label types
    and at least one continous feature
    Checks how many samples are inside a given radius

    Arguments:
        
    """
    k = 6
    nx = np.ones(X.shape[0]) * k
    ny = np.ones(X.shape[0]) * k
    nx[:] = np.nan
    ny[:] = np.nan

    D_x = KNN.get_dist_matrix(X, f_types)
    D_x.sort()

    new_y = assert_df(y)
    new_types = assert_series(l_type)
    D_y = KNN.get_dist_matrix(new_y, new_types)
    D_y.sort()

    for row in range(X.shape[0]):
        # Get distances inside features and labels
        dist_x = D_x[row, :]
        dist_y = D_y[row, :]

        # Update statistics if sample contains non-nan values
        radius = max(dist_x[k + 1], dist_y[k + 1])
        if not np.isinf(radius):
            nx[row] = (dist_x <= radius).sum() - 1
            ny[row] = (dist_y <= radius).sum() - 1

    nx = nx[~np.isnan(nx)]
    ny = ny[~np.isnan(ny)]

    mi = digamma(len(nx)) + digamma(k) - (1 / k) - \
        digamma(np.mean(nx)) - digamma(np.mean(ny))
    return max(mi, 0)


def _get_mi_cd(X, y, f_types):
    """
    Estimate mutual information for discrete label types
    and at least one continous feature
    Checks how many samples of the same type are inside a given radius

    Arguments:
        
    """
    k = 6
    n = np.ones(X.shape[0]) * k
    m = np.ones(X.shape[0]) * k
    n[:] = np.nan
    m[:] = np.nan

    D = KNN.get_dist_matrix(X, f_types)

    for row in range(X.shape[0]):
        # Get radius for k nearest neighbors within same class
        dist_cond = D[row, y == y[row]]
        dist_cond.sort()
        max_k = min(k + 1, len(dist_cond) - 1)
        radius = dist_cond[max_k]

        # Get distances for all samples
        dist_full = D[row, :]

        # Update statistics if sample contains non-nan values
        if not np.isinf(radius):
            m[row] = (dist_full <= radius).sum() - 1
            n[row] = len(dist_cond)

    m = m[~np.isnan(m)]
    n = n[~np.isnan(n)]

    mi = digamma(len(m)) - np.mean(digamma(n)) + \
        digamma(k) - np.mean(digamma(m))
    return max(mi, 0)


def _get_mi_dd(X, y):
    """
    Estimate MI between discrete feature and discrete label
    Only works for 1-d features and creates class for missing values

    Arguments:

    """
    mi = mutual_info_score(X, y)
    return max(mi, 0)


def get_mutual_information(X, y, f_types, l_type):
    """
    Estimate mutual information using different approaches

    Arguments:

    """
    # We do not have an estimator which takes multiple d's as X
    # We also do not have an estimator for d -> c
    # Split up features when only nominal features are present
    # Inverse data when target is continous (assumes symmetric estimation)

    ### CASE DD - C/D ###
    if "numeric" not in f_types.values:
        mi_s = np.zeros(X.shape[1])
        for i, col in enumerate(X):
            if l_type == "nominal":
                ### CASE D - D ###
                mi_s[i] = _get_mi_dd(X[col], y)
            else:
                ### CASE D - C ###
                new_X = assert_df(y)
                new_X.columns = [y.name]
                new_types = assert_types(l_type, y.name)
                mi_s[i] = _get_mi_cd(new_X, y, new_types)
        return np.mean(mi_s)

    # Use standard estimators when numerical features are present
    ### CASE CC[D] - C/D ###
    if l_type == "nominal":
        return _get_mi_cd(X, y, f_types)
    else:
        return _get_mi_cc(X, y, f_types, l_type)

    sys.exit("MI messed up types")
    return -1
