"""
    Tree Classifier class using Orange implementation
"""
import pandas as pd
from Orange.data import Table, Domain, DiscreteVariable
from Orange.classification import TreeLearner as TreeClassifier
from Orange.regression import TreeLearner as TreeRegressor

from project.utils import assert_series, assert_df


class Tree():
    def __init__(self, domain):
        """
        Class which predicts label for unseen samples

        Arguments:
            domain {Domain} -- Orange domain
        """
        self.domain = domain

    def fit(self, X, y):
        """
        Fit the tree classifier

        Arguments:
            X {[df]} -- Dataframe containing the features
            y {pd.series} -- Label vector
        """

        X = assert_df(X)
        y = assert_series(y)
        self.attributes = [
            a for a in self.domain.attributes if a.name in X.columns.values
        ]
        self.columns = [a.name for a in self.attributes]

        s_domain = Domain(self.attributes, class_vars=self.domain.class_var)
        rows = pd.concat([X[self.columns], y], axis=1).values.tolist()
        train = Table.from_list(domain=s_domain, rows=rows)

        if isinstance(self.domain.class_var, DiscreteVariable):
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
        test = Table.from_list(domain, X[self.columns].values.tolist())

        predictions = self.tree(test.X)
        if isinstance(self.domain.class_var, DiscreteVariable):
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
            "domain": self.domain,
        }
