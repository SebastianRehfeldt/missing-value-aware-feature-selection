from copy import deepcopy
import numpy as np
import pandas as pd
from project.utils.assertions import assert_data, assert_df, assert_types, assert_series
from Orange.data.variable import DiscreteVariable, ContinuousVariable
from Orange.data import Domain, Table


class Data():

    def __init__(self, X, y, f_types, l_type, shape):
        self.X = X
        self.y = y
        self.f_types = f_types
        self.l_type = l_type
        self.shape = shape

        assert_data(self)

    def replace(self, **kwargs):
        new_data = deepcopy(self)

        for k, v in kwargs.items():
            setattr(new_data, k, v)

        return assert_data(new_data)

    def get(self, subspace):
        # TODO check if works for 1d subspace
        new_X = assert_df(self.X[subspace])
        new_types = assert_types(self.f_types[subspace], subspace)
        return new_X, new_types

    def select(self, subspace):
        new_X, types = self.get(subspace)
        return self.replace(X=new_X, f_types=types, shape=new_X.shape)

    def inverse(self):
        """
        Replaces feature matrix with label vector
        """
        new_X = assert_df(self.y)
        new_X.columns = [self.y.name]

        new_types = assert_types(self.l_type, self.y.name)
        return self.replace(X=new_X, f_type=new_types)

    def select_inverse(self, col):
        """
        Select exactly one column and flip X and y

        Arguments:
            col {str} -- Column name
        """
        new_X = assert_df(self.y)
        new_X.columns = [self.y.name]

        new_y = assert_series(self.X[col])
        new_y.rename(col)

        f_types = assert_types(self.l_type, self.y.name)
        l_type = self.f_types[col]

        inversed = Data(new_X, new_y, f_types, l_type, new_X.shape)
        return assert_data(inversed)

    def to_table(self):
        f_array = []
        for i, t in enumerate(self.f_types):
            f_name = self.X.columns[i]
            if t == "numeric":
                f_array.append(ContinuousVariable(f_name))
            else:
                values = [v for v in np.unique(self.X[f_name])]
                f_array.append(DiscreteVariable(f_name, values=values))

        l_name = self.y.name
        if self.l_type == "nominal":
            values = [v for v in np.unique(self.y)]
            class_var = DiscreteVariable(f_name, values=values)
        else:
            class_var = ContinuousVariable(l_name)

        combined = pd.concat([self.X, self.y], axis=1)
        domain = Domain(f_array, class_vars=class_var)
        return Table.from_list(domain=domain, rows=combined.values.tolist())
