"""
    Base class for subspacing approaches.
"""
import itertools
from abc import abstractmethod
from project.shared.selector import Selector
import numpy as np


class Subspacing(Selector):
    def __init__(self, f_types, l_type, shape, **kwargs):
        """
        Base class for subspacing approaches.

        Arguments:
            f_types {pd.Series} -- Series containing feature types
            l_type {str} -- Type of label
            shape {tuple} -- Tuple containing the shape of features
        """
        super().__init__(f_types, l_type, shape, **kwargs)

    @abstractmethod
    def _init_parameters(self, parameters):
        """
        Init parameters for subspacing approaches

        Arguments:
            parameters {dict} -- Parameter dict for subspacing
        """
        self.params = {}
        raise NotImplementedError("subclasses must override _init_parameters")

    @abstractmethod
    def _evaluate_subspace(self, X, types):
        """
        Evaluate a feature subspace and return results as dict

        Arguments:
            X {df} -- Dataframe containing features
            types {pd.series} -- Series containing the feature types
        """
        raise NotImplementedError(
            "subclasses must implement _evaluate_subspace")

    @abstractmethod
    def _deduce_feature_importances(self, score_map):
        """
        Deduce feature importances from subset evaluations

        Arguments:
            score_map {list} -- List of subspace evaluations
        """
        raise NotImplementedError(
            "subclasses must implement _deduce_feature_importances")

    def _get_unique_subscapes(self):
        """
        Return unique feature subspaces
        """
        subspaces = [None] * self.params["n"]
        names = self.data.X.columns
        for i in range(self.params["n"]):
            m = np.random.randint(0, self.params["m"], 1)[0] + 1
            f = list(np.random.choice(names, m, replace=False))
            subspaces[i] = sorted(f)

        subspaces.sort()
        return list(subspaces for subspaces, _ in itertools.groupby(subspaces))

    def _evaluate_subspaces(self, subspaces):
        """
        Collect and return subset evaluations

        Arguments:
            subspaces {list} -- List of feature subspaces
        """
        knowledgebase = []
        for subspace in subspaces:
            features, types = self.data.get_subspace(subspace)
            score = self._evaluate_subspace(features, types)
            knowledgebase.append({"features": subspace, "score": score})
        return knowledgebase
