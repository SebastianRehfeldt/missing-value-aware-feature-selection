"""
    Utils class for estimating mutual information using PDS
"""
import sys
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.metrics import mutual_info_score
from project import Data
from project.shared.neighbors import Neighbors
from project.utils.assertions import assert_data, assert_df, assert_types


def _get_mi_cc(data):
    """
    Estimate mutual information for continous label types and at least one continous feature
    Checks how many samples are inside a given radius

    Arguments:
        data {data} -- Data Object for estimation
    """
    new_data = data._replace(
        features=data.labels.to_frame(), f_types=pd.Series(data.l_type))
    new_data = assert_data(new_data)

    # Create Neighbor objects for features and labels
    nn_x = Neighbors(data)
    nn_y = Neighbors(new_data)

    k = 6
    nx = np.ones(data.shape[0]) * k
    ny = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        # Get distances inside features
        sample_x = data.features.iloc[i]
        dist_x = nn_x.partial_distances(sample_x)
        dist_x.sort()

        # Get distances inside labels
        sample_y = new_data.features.iloc[i]
        dist_y = nn_y.partial_distances(sample_y)
        dist_y.sort()

        # Update statistics if sample contains non-nan values
        max_dist = max(dist_x[k+1], dist_y[k+1])
        if not np.isinf(max_dist):
            nx[i] = (dist_x <= max_dist).sum() - 1
            ny[i] = (dist_y <= max_dist).sum() - 1

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
    # Create neighbors object for all samples
    nn = Neighbors(data)
    k = 6
    n = np.ones(data.shape[0]) * k
    m = np.ones(data.shape[0]) * k
    for i in range(data.shape[0]):
        # Create neighbors object for samples of same class
        label = data.labels[i]
        features = data.features[data.labels == label].reset_index(drop=True)
        labels = data.labels[data.labels == label].reset_index(drop=True)
        new_data = data._replace(
            features=features, labels=labels, shape=features.shape)
        new_data = assert_data(new_data)
        nn_cond = Neighbors(new_data)

        # Get radius for k nearest neighbors
        sample = data.features.iloc[i]
        dist_cond = nn_cond.partial_distances(sample)
        dist_cond.sort()
        max_k = min(k+1, len(dist_cond) - 1)
        max_dist = dist_cond[max_k]

        # Get distances for all samples
        dist_full = nn.partial_distances(sample)

        # Update statistics if sample contains non-nan values
        if not np.isinf(max_dist):
            m[i] = (dist_full <= max_dist).sum() - 1
            n[i] = new_data.features.shape[0]

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
    mi = mutual_info_score(data.features.iloc[:, 0], data.labels)
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
            for i, col in enumerate(data.features):
                features = assert_df(data.features[col])
                types = assert_types(data.f_types[col], col)
                selected_data = data._replace(
                    features=features, f_types=types, shape=features.shape)

                selected_data = assert_data(selected_data)
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
                # TODO utils function
                mi_s = np.zeros(data.shape[1])
                for i, col in enumerate(data.features):
                    new_X = data.labels.to_frame()
                    new_X.columns = [col]
                    new_y = data.features[col]
                    f_types = assert_types(data.l_type, col)
                    l_type = data.f_types[col]

                    inversed_data = Data(
                        new_X, new_y, f_types, l_type, new_X.shape)
                    inversed_data = assert_data(inversed_data)
                    mi_s[i] = _get_mi_cd(inversed_data)
                return np.mean(mi_s)

    print(data)
    sys.exit("MI messed up types")
    return -1
