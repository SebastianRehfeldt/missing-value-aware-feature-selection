import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class SKClassifier():
    def __init__(self, f_types, kind="svm", **kwargs):
        self.f_types = f_types
        self.kind = kind
        self.params = {}

        self.fill_value = 42 if kind == "tree" else 0

    def encode(self, X):
        nans = X == "?"
        for col in X:
            if self.f_types[col] == "nominal":
                X[col] = LabelEncoder().fit_transform(X[col])
        X[nans] = np.nan
        return X

    def add_dummies(self, X):
        nans = np.isnan(X)
        nans.columns = [n + "_isnan" for n in nans.columns]
        nans = nans.astype(int)

        new_X = pd.concat([X, nans], axis=1)
        return new_X.fillna(0)

    def fit(self, X_train, y):
        X = self.encode(deepcopy(X_train))
        X = X.fillna(self.fill_value)

        self.clf = {
            "svm": SVC,
            "bayes": GaussianNB,
            "xgb": XGBClassifier,
            "knn": KNeighborsClassifier,
            "logreg": LogisticRegression,
            "tree": DecisionTreeClassifier,
        }[self.kind]()
        self.clf.fit(X, y)
        return self

    def predict(self, X_test):
        X = self.encode(deepcopy(X_test))
        X = X.fillna(self.fill_value)
        return self.clf.predict(X)

    def get_params(self, deep=False):
        return {
            "f_types": self.f_types,
            "kind": self.kind,
            **self.params,
        }
