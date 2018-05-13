"""
    Tree Classifier class using Orange implementation
"""
import pandas as pd
from Orange.data import Table, Domain
from Orange.classification import TreeLearner as TreeClassifier
from Orange.regression import TreeLearner as TreeRegressor
from project import Data
from project.utils.assertions import assert_series, assert_data, assert_l_type, assert_df, assert_types


class Tree():

    def __init__(self, data, **kwargs):
        """
        Class which predicts label for unseen samples

        Arguments:
            data {Data} -- Data Object
        """
        self.data = data
        self.domain = data.to_table().domain

    def fit(self, X, y):
        """
        Fit the tree classifier

        Arguments:
            X {[df]} -- Dataframe containing the features
            y {pd.series} -- Label vector
        """

        self.attributes = [
            a for a in self.domain.attributes if a.name in X.columns.values]
        s_domain = Domain(self.attributes, class_vars=self.domain.class_var)

        rows = pd.concat([X, y], axis=1).values.tolist()
        train = Table.from_list(domain=s_domain, rows=rows)

        if self.data.l_type == "nominal":
            self.tree = TreeClassifier().fit_storage(train)
        else:
            self.tree = TreeRegressor().fit_storage(train)
        return self

    def predict(self, X):
        """
        Make prediction for unseen samples

        Arguments:
            X {[df]} -- Dataframe containing the features
        """
        X = assert_df(X)
        domain = Domain(list(self.attributes))
        test = Table.from_list(domain, X.values.tolist())

        predictions = self.tree(test.X)
        if self.data.l_type == "nominal":
            labels = self.domain.class_var.values
            predictions = [labels[pred] for pred in predictions]
        return predictions

    def get_params(self, deep=False):
        """
        Return params

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "data": self.data,
        }
