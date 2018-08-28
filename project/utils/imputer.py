import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fancyimpute import KNN, MICE, MatrixFactorization, SimpleFill, SoftImpute
from project.utils import assert_data


class Imputer():
    def __init__(self, f_types, strategy="knn"):
        warnings.filterwarnings(
            module='sklearn*', action='ignore', category=DeprecationWarning)
        warnings.filterwarnings(
            module='fancy*', action='ignore', category=RuntimeWarning)
        self.f_types = f_types
        self.strategy = strategy

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        cols = self.f_types[X.columns].index.tolist()
        return self._complete(pd.DataFrame(X, columns=cols), cols)

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X, y)

    def get_params(self, deep=False):
        return {
            "f_types": self.f_types,
            "strategy": self.strategy,
        }

    def _encode(self, X, nom_cols, nans):
        self.encoders, self.modes = {}, {}
        for col in nom_cols:
            # replace "?" with mode so that labelencoder does not fills with ?
            elements = X[col].value_counts()
            mode = elements.index[0]
            if mode == "?":
                mode = elements.index[1]
            X[col] = X[col].where(X[col] != "?", mode)
            self.modes[col] = mode

            # encode
            le = LabelEncoder().fit(X[col])
            X[col] = le.transform(X[col])
            self.encoders[col] = le

        self.scaler = StandardScaler().fit(X[nom_cols])
        X[nom_cols] = self.scaler.transform(X[nom_cols])
        X[nom_cols] += 1e-8
        X[nans] = np.nan
        return X

    def _decode(self, X, nom_cols, nans):
        failed = X[nom_cols] == 0
        X[nom_cols] = self.scaler.inverse_transform(X[nom_cols])
        X[nans] = np.round(X[nans], 0)
        X[nom_cols] = X[nom_cols].astype(int)
        for col in nom_cols:
            classes = self.encoders[col].classes_
            X[col].clip(0, len(classes) - 1, inplace=True)
            X[col] = self.encoders[col].inverse_transform(X[col])
            X[col].values[failed[col]] = self.modes[col]
        return X

    def _complete(self, X, cols=None):
        types = self.f_types if cols is None else self.f_types[cols]
        num_cols = types.loc[self.f_types == "numeric"].index
        nom_cols = types.loc[self.f_types == "nominal"].index

        has_nominal_nans = False
        if len(nom_cols) > 0:
            nans = X == "?"
            has_nominal_nans = nans.values.any()

        if not has_nominal_nans and X[num_cols].notnull().values.all():
            return X

        X_copy = X.copy()
        if len(nom_cols) > 0:
            X_copy = self._encode(X_copy, nom_cols, nans)

        X_copy[:] = self._get_imputer().complete(X_copy)

        if len(nom_cols) > 0:
            X_copy = self._decode(X_copy, nom_cols, nans)
        return X_copy

    def complete(self, data):
        data = assert_data(data)
        complete_features = self._complete(data.X)
        return data.replace(copy=True, X=complete_features)

    def _get_imputer(self):
        if self.strategy == "simple":
            return SimpleFill()

        return {
            "knn": KNN,
            "mice": MICE,
            "matrix": MatrixFactorization,
            "soft": SoftImpute,
        }[self.strategy](verbose=False)
