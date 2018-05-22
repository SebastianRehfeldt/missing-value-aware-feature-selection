from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error

from project.classifier import KNN, Tree
from project.shared import get_mutual_information


def evaluate_subspace(X, y, f_types, l_type, domain, **params):
    method = params.get("eval_method", "mi")
    return {
        "mi": _evaluate_subspace_mi,
        "knn": _evaluate_subspace_clf,
        "tree": _evaluate_subspace_clf,
    }[method](X, y, f_types, l_type, domain, **params)


def _evaluate_subspace_mi(X, y, f_types, l_type, domain, **params):
    """
    Evaluate a subspace using mi

    Arguments:
    """
    return get_mutual_information(X, y, f_types, l_type, **params)


def _evaluate_subspace_clf(X, y, f_types, l_type, domain, **params):
    """
    Evaluate a subspace using knn

    Arguments:
    """
    if params["eval_method"] == "knn":
        clf = KNN(f_types, l_type, **params)
    else:
        clf = Tree(domain)

    scoring, stratify = "accuracy", y
    if l_type == "numeric":
        scoring, stratify = "neg_mean_squared_error", None

    if params["use_cv"]:
        cv = StratifiedKFold(y, n_folds=3, shuffle=True)
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        return np.mean(scores)
    else:
        split = train_test_split(X, y, test_size=0.5, stratify=stratify)
        X_train, X_test, y_train, y_test = split
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return {
            "accuracy": accuracy_score,
            "neg_mean_squared_error": mean_squared_error,
        }[scoring](y_test, y_pred)
