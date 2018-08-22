from copy import deepcopy
from sklearn.pipeline import Pipeline

from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.rar.rar import RaR
from project.classifier import KNN, Tree, SKClassifier
from project.utils.imputer import Imputer


def get_selectors(data, complete, names, max_k=None):
    d = [data.f_types, data.l_type, data.shape]
    d2 = [complete.f_types, complete.l_type, complete.shape]
    max_k = data.shape[1] if max_k is None else max_k

    selectors = {
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

    return [selectors[name] for name in names]


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


def swap_pipeline_steps(pipe):
    temp_step = deepcopy(pipe.steps[0])
    pipe.steps[0] = deepcopy(pipe.steps[1])
    pipe.steps[1] = temp_step


def get_pipelines(data, k, names, classifier):
    d = [data.f_types, data.l_type, data.shape]

    selectors = {
        "rar": RaR(
            *d, k=k, alpha=0.02, contrast_iterations=250, n_subspace=800),
        "rknn": RKNN(*d, k=k),
        "sfs": SFS(*d, k=k, do_stop=True, eval_method="tree"),
        "mi": Filter(*d, k=k),
        "relief_sk": Ranking(*d, k=k, eval_method="myrelief"),
        "fcbf_sk": Ranking(*d, k=k, eval_method="fcbf"),
        "mrmr": Ranking(*d, k=k, eval_method="mrmr"),
        "cfs": Ranking(*d, k=k, eval_method="cfs"),
        "relief_o": Orange(*d, k=k, eval_method="relief"),
        "fcbf_o": Orange(*d, k=k, eval_method="fcbf"),
        "rf": Orange(*d, k=k, eval_method="rf"),
        "xgb": Embedded(*d, k=k),
    }

    clf = get_classifiers(data, [classifier])[0]

    # DEFINE PIEPLINES
    pipelines = []
    for name in names:
        if name == "complete":
            pipelines.append(Pipeline(steps=[('classify', clf)]))
        if "+ impute" in name:
            strategy, selector = name.split(" ")[-1], name.split(" ")[0]
            pipelines.append(
                Pipeline(steps=[
                    ('reduce', deepcopy(selectors[selector])),
                    ('imputer', Imputer(data.f_types, strategy)),
                    ('classify', clf),
                ]))
        elif "impute +" in name:
            strategy, selector = name.split(" ")[0], name.split(" ")[-1]
            pipelines.append(
                Pipeline(steps=[
                    ('imputer', Imputer(data.f_types, strategy)),
                    ('reduce', deepcopy(selectors[selector])),
                    ('classify', clf),
                ]))
        else:
            pipelines.append(
                Pipeline(steps=[
                    ('reduce', deepcopy(selectors[name])),
                    ('classify', clf),
                ]))

    return pipelines
