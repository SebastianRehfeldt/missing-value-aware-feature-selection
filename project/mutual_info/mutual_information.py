"""
    Utils class for estimating mutual information using PDS
"""
import sys
import numpy as np
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
from project.utils import assert_df, assert_series

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

    mi = digamma(X.shape[0]) + digamma(k) - (1 / k) - \
        digamma(np.mean(nx)) - digamma(np.mean(ny))
    return max(mi, 0)


def _get_mi_cd(X, y, f_types, l_type):
    """
    Estimate mutual information for discrete label types
    and at least one continous feature
    Checks how many samples of the same type are inside a given radius

    Arguments:
        
    """
    k = 6
    n = np.ones(X.shape[0]) * k
    m = np.ones(X.shape[0]) * k

    rn = np.random.randn(*X.shape)
    noise = 1e-10 * X.abs().mean()[0] * rn
    X_salted = X + noise
    D = KNN.get_dist_matrix(X_salted, f_types)

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
        else:
            m[row] = np.nan
            n[row] = np.nan

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
    # Estimate MI for nominal targets
    if l_type == "nominal":
        # When numerical features are present, distances can be calculated
        if "numeric" in f_types.values:
            return _get_mi_cd(X, y, f_types, l_type)

        # Estimate MI for each nominal feature and return mean
        mi_s = np.zeros(X.shape[1])
        for i, col in enumerate(X):
            # TODO
            # selected_data = data.select(col)
            mi_s[i] = _get_mi_dd(X, y)
        return np.mean(mi_s)

    # Estimate MI for numerical targets
    else:
        # When numerical features are present, distances can be calculated
        if "numeric" in f_types.values:
            return _get_mi_cc(X, y, f_types, l_type)
        else:
            # Estimate MI for each nominal feature and return mean
            # Features and labels are flipped to match get_mi_cd
            # TODO check symmilarity assumption for mi estimation
            mi_s = np.zeros(X.shape[1])
            for i, col in enumerate(X):
                # TODO
                # inversed_data = data.select_inverse(col)
                mi_s[i] = _get_mi_cd(X, y, f_types, l_type)
            return np.mean(mi_s)

    sys.exit("MI messed up types")
    return -1
