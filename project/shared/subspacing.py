"""
    Base class for subspacing approaches.
"""
import itertools
from abc import ABC, abstractmethod
import numpy as np
from project.utils.assertions import assert_df


class Subspacing(ABC):
    def __init__(self, data, **kwargs):
        """
        Base class for subspacing approaches.

        Arguments:
            data {Data} -- Data object
        """
        self.data = data
        self._init_parameters(kwargs)

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
    def _evaluate_subspace(self, subspace, features):
        """
        Evaluate a feature subspace and return results as dict

        Arguments:
            subspace {list} -- List of features in subspace
            features {df} -- Dataframe containing features
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
        names = self.data.features.columns
        for i in range(self.params["n"]):
            m = np.random.randint(0, self.params["m"], 1)[0] + 1
            print(m)
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
            # TODO check if works for 1d subspace
            features = self.data.features[subspace]
            features = assert_df(features)

            knowledgebase.append({
                "features": subspace,
                "score": self._evaluate_subspace(subspace, features)
            })
        return knowledgebase
