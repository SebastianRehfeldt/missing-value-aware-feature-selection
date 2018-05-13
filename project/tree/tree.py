"""
    Tree Classifier class using Orange implementation
"""
from Orange.data import Table, Domain
from Orange.classification import TreeLearner as TreeClassifier
from Orange.regression import TreeLearner as TreeRegressor
from project import Data
from project.utils.assertions import assert_series, assert_data, assert_l_type, assert_df, assert_types


class Tree():

    def __init__(self, f_types, l_type, **kwargs):
        """
        Class which predicts label for unseen samples

        Arguments:
            f_types {pd.series} -- Series of feature types
            l_type {str} -- Type of label
        """
        self.f_types = assert_series(f_types)
        self.l_type = assert_l_type(l_type)

    def fit(self, X, y):
        """
        Fit the tree classifier

        Arguments:
            X {[df]} -- Dataframe containing the features
            y {pd.series} -- Label vector
        """
        types = assert_types(self.f_types[X.columns.values], X.columns.values)
        data = Data(X, y, types, self.l_type, X.shape).to_table()
        if self.l_type == "nominal":
            self.labels = data.domain.class_var.values
            self.tree = TreeClassifier().fit_storage(data)
        else:
            self.domain = data.domain
            self.tree = TreeRegressor().fit_storage(data)
        return self

    def predict(self, X):
        """
        Make prediction for unseen samples

        Arguments:
            X {[df]} -- Dataframe containing the features
        """
        X = assert_df(X)
        if self.l_type == "nominal":
            predictions = self.tree(X.values)
            predictions = [self.labels[pred] for pred in predictions]
        else:
            domain = Domain(list(self.domain.attributes))
            test = Table.from_list(domain, X.values.tolist())
            predictions = self.tree(test.X)
        return predictions

    def get_params(self, deep=False):
        """
        Return params

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "f_types": self.f_types,
            "l_type": self.l_type,
        }
