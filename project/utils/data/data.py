import numpy as np
import pandas as pd
from copy import deepcopy
from Orange.data.variable import DiscreteVariable, ContinuousVariable
from Orange.data import Domain, Table
from sklearn.model_selection import StratifiedKFold

from .assertions import assert_data, assert_df, assert_types


class Data():
    def __init__(self, X, y, f_types, l_type, shape):
        self.X = X
        self.y = y
        self.f_types = f_types
        self.l_type = l_type
        self.shape = shape

        assert_data(self)

    def replace(self, copy=False, **kwargs):
        new_data = deepcopy(self) if copy else self

        for k, v in kwargs.items():
            setattr(new_data, k, v)

        return assert_data(new_data)

    def get_subspace(self, subspace):
        new_X = assert_df(self.X[subspace])
        new_types = assert_types(self.f_types[subspace], subspace)
        return new_X, new_types

    def _add_salt_X(self):
        cols = self.f_types.loc[self.f_types == "numeric"].index
        if len(cols) > 0:
            rn = np.random.randn(self.X.shape[0], len(cols))
            noise = 1e-10 * self.X[cols].abs().mean()[0] * rn
            salted = self.X.copy()
            salted[cols] = self.X[cols] + noise
            return salted
        return self.X

    def _add_salt_y(self):
        if self.l_type == "numeric":
            rn = np.random.randn(len(self.y))
            noise = 1e-10 * self.y.abs().mean() * rn
            return self.y + noise
        return self.y

    def add_salt(self, copy=False):
        X_salted = self._add_salt_X()
        y_salted = self._add_salt_y()
        return self.replace(copy=copy, X=X_salted, y=y_salted)

    def shuffle_rows(self, copy=False, seed=None):
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(self.shape[0])
        X_shuffled = self.X.iloc[indices].reset_index(drop=True)
        y_shuffled = self.y.iloc[indices].reset_index(drop=True)
        return self.replace(copy=copy, X=X_shuffled, y=y_shuffled)

    def to_table(self):
        attributes = [
            Data.get_variable(c_type, self.X.columns[i], self.X.iloc[:, i])
            for i, c_type in enumerate(self.f_types)
        ]
        class_var = Data.get_variable(self.l_type, self.y.name, self.y)

        combined = pd.concat([self.X, self.y], axis=1)
        domain = Domain(attributes, class_vars=class_var)
        return Table.from_list(domain=domain, rows=combined.values.tolist())

    def split(self, n_splits=3):
        splits = []
        kf = StratifiedKFold(n_splits)
        f_types, l_type = self.f_types, self.l_type

        for train_index, test_index in kf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            train = Data(X_train, y_train, f_types, l_type, X_train.shape)
            test = Data(X_test, y_test, f_types, l_type, X_test.shape)
            splits.append((train, test))
        return splits

    @staticmethod
    def get_variable(c_type, name, column):
        if c_type == "nominal":
            values = [v for v in np.unique(column)]
            return DiscreteVariable(name, values=values)
        else:
            return ContinuousVariable(name)
