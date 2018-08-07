from scipy.spatial.distance import cdist, pdist, squareform
from .partial_distance import partial_distance, partial_distance_orange


def get_dist_matrix(XA, f_types, XB=None, **kwargs):
    dist_params = {
        "nominal_distance": kwargs.get("nominal_distance", 1),
        "f_types": f_types.values,
    }
    metric = kwargs.get("distance_metric", "partial")
    distance = {
        "partial": partial_distance,
        "orange": partial_distance_orange,
    }[metric]

    if XB is None:
        D = pdist(XA, metric=distance, **dist_params)
        D = squareform(D)
    else:
        D = cdist(XB, XA, metric=distance, **dist_params)
    return D
