from copy import deepcopy

from project.rar.rar import RaR
from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.baseline import Baseline
from project.classifier import KNN, Tree, SKClassifier


def get_selectors(data, complete, names, max_k=None):
    d = [data.f_types, data.l_type, data.shape]
    d2 = [complete.f_types, complete.l_type, complete.shape]
    max_k = data.shape[1] if max_k is None else max_k

    selectors = {
        "baseline": Baseline(*d),
        "rar_del": RaR(*d, approach="deletion"),
        "rar_fuz": RaR(*d, approach="fuzzy", weight_approach="imputed"),
        "rknn": RKNN(*d),
        "sfs": SFS(*d2, k=max_k, do_stop=True, eval_method="tree"),
        "mi": Filter(*d),
        "relief_sk": Ranking(*d, eval_method="myrelief"),
        "fcbf_sk": Ranking(*d, eval_method="fcbf"),
        "mrmr": Ranking(*d, eval_method="mrmr"),
        "cfs": Ranking(*d, eval_method="cfs"),
        "relief_o": Orange(*d, eval_method="relief"),
        "fcbf_o": Orange(*d, eval_method="fcbf"),
        "rf": Orange(*d, eval_method="rf"),
        "xgb": Embedded(*d),
    }

    return [deepcopy(selectors[name]) for name in names]


def get_classifiers(data, complete, names):
    classifiers = {
        "knn": KNN(data.f_types, data.l_type, knn_neighbors=6),
        "tree": Tree(complete.to_table().domain),
        "svm": SKClassifier(data.f_types, "svm"),
        "gnb": SKClassifier(data.f_types, "gnb"),
        "xgb": SKClassifier(data.f_types, "xgb"),
        "log": SKClassifier(data.f_types, "log"),
        "sk_knn": SKClassifier(data.f_types, "knn"),
        "sk_tree": SKClassifier(data.f_types, "tree"),
    }
    return [classifiers[name] for name in names]
