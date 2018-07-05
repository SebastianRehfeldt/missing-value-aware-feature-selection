from scipy.spatial.distance import cdist, pdist, squareform
from .partial_distance import partial_distance


def get_dist_matrix(XA, f_types, XB=None, **kwargs):
    dist_params = {
        "nominal_distance": kwargs.get("nominal_distance", 1),
        "f_types": f_types.values,
    }
    if XB is None:
        D = pdist(XA, metric=partial_distance, **dist_params)
        D = squareform(D)
    else:
        D = cdist(XB, XA, metric=partial_distance, **dist_params)
    return D
