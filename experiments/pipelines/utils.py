from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.rar.rar import RaR
from project.classifier import KNN, Tree
from project.utils.imputer import Imputer


def get_pipelines(data, k, names, classifier):
    d = [data.f_types, data.l_type, data.shape]

    selectors = {
        "rar": RaR(
            *d, k=k, alpha=0.02, contrast_iterations=250, n_subspace=800),
        "rknn": RKNN(*d, k=k),
        "sfs": SFS(*d, k=k, do_stop=True, eval_method="tree"),
        "mi": Filter(*d, k=k),
        "relief_sk": Ranking(*d, eval_method="myrelief"),
        "fcbf_sk": Ranking(*d, eval_method="fcbf"),
        "mrmr": Ranking(*d, eval_method="mrmr"),
        "cfs": Ranking(*d, eval_method="cfs"),
        "relief_o": Orange(*d, eval_method="relief"),
        "fcbf_o": Orange(*d, eval_method="fcbf"),
        "rf": Orange(*d, eval_method="rf"),
    }

    clf = {
        "knn": KNN(data.f_types, data.l_type, knn_neighbors=6),
        "tree": Tree(data.to_table().domain),
        "gnb": GaussianNB(),
    }[classifier]

    # DEFINE PIEPLINES
    pipelines = []
    for name in names:
        if name == "complete":
            pipelines.append(Pipeline(steps=[('classify', clf)]))
        if "+ impute" in name:
            print("create imputer pipe")
            pipelines.append(
                Pipeline(steps=[
                    ('reduce', selectors[name.split(" ")[0]]),
                    ('imputer', Imputer(data.f_types, "simple")),
                    ('classify', clf),
                ]))
        else:
            pipelines.append(
                Pipeline(steps=[
                    ('reduce', selectors[name]),
                    ('classify', clf),
                ]))

    return pipelines
