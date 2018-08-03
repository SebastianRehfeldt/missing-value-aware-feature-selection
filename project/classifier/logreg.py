import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression as skl_logreg


class LogReg():
    def __init__(self, f_types, **kwargs):
        self.f_types = f_types
        self.params = {}

    def encode(self, X):
        for col in X:
            if self.f_types[col] == "nominal":
                # encode
                X[col] = LabelEncoder().fit_transform(X[col])
        return X

    def add_dummies(self, X):
        nans = np.isnan(X)
        nans.columns = [n + "_isnan" for n in nans.columns]
        nans = nans.astype(int)

        new_X = pd.concat([X, nans], axis=1)
        return new_X.fillna(0)

    def fit(self, X, y):
        X = self.encode(X)
        X = self.add_dummies(X)
        self.clf = skl_logreg(C=1e5).fit(X, y)
        return self

    def predict(self, X_test):
        X_test = self.encode(X_test)
        X_test = self.add_dummies(X_test)
        return self.clf.predict(X_test)

    def get_params(self, deep=False):
        return {
            "f_types": self.f_types,
            **self.params,
        }
