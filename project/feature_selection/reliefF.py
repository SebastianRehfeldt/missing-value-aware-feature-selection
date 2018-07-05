"""
    Original implementation comes from skfeature package
"""
import numpy as np
from skfeature.utility.util import reverse_argsort
from project.shared.partial_distance import partial_distance
from scipy.spatial.distance import cdist


def reliefF(X, y, dist_params, mode="rank", **kwargs):
    """
    This function implements the reliefF feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        parameters of reliefF:
        k: {int}
            choices for the number of neighbors (default k = 5)

    Output
    ------
    score: {numpy array}, shape (n_features,)
        reliefF score for each feature

    Reference
    ---------
    Robnik-Sikonja, Marko et al. "Theoretical and empirical analysis of relieff and rrelieff." Machine Learning 2003.
    Zhao, Zheng et al. "On Similarity Preserving Feature Selection." TKDE 2013.
    """

    def feature_ranking(score):
        """
        Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
        feature is
        """
        idx = np.argsort(score, 0)
        return idx[::-1]

    if "k" not in list(kwargs.keys()):
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape

    # calculate pairwise distances between instances
    distance = cdist(X, X, metric=partial_distance, **dist_params)

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        c = np.unique(y).tolist()

        stop_dict = dict()
        for label in c:
            stop_dict[label] = 0
        del c[c.index(y[idx])]

        p_dict = dict()
        p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)

        for label in c:
            p_label_c = float(len(y[y == label])) / float(n_samples)
            p_dict[label] = p_label_c / (1 - p_label_idx)
            near_miss[label] = []

        distance_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            distance_sort.append([distance[idx, i], int(i), y[i]])
        distance_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if distance_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(distance_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[distance_sort[i][2]]) < k:
                    near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                else:
                    if len(near_miss[distance_sort[i][2]]) == k:
                        stop_dict[distance_sort[i][2]] = 1
            stop = True
            for (key, value) in list(stop_dict.items()):
                if value != 1:
                    stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            dist = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                if isinstance(self_fea[i], str):
                    dist[i] = 0 if self_fea[i] == X[ele, i] else 1
                else:
                    dist[i] = abs(self_fea[i] - X[ele, i])

            near_hit_term += dist

        near_miss_term = dict()
        for (label, miss_list) in list(near_miss.items()):
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                dist = np.zeros(X.shape[1])
                for i in range(X.shape[1]):
                    if isinstance(self_fea[i], str):
                        dist[i] = 0 if self_fea[i] == X[ele, i] else 1
                    else:
                        dist[i] = abs(self_fea[i] - X[ele, i])
                near_miss_term[label] = dist + np.array(near_miss_term[label])
            score += near_miss_term[label] / (k * p_dict[label])
        score -= near_hit_term / k
    if mode == 'raw':
        return score
    elif mode == 'index':
        return feature_ranking(score)
    elif mode == 'rank':
        return reverse_argsort(feature_ranking(score), X.shape[1])
