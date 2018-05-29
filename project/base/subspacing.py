"""
    Base class for subspacing approaches.
"""
import itertools
from abc import abstractmethod
from project.base import Selector
import numpy as np
from multiprocessing import Pool


class Subspacing(Selector):
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

    def _init_parameters(self, **kwargs):
        super()._init_parameters(**kwargs)
        # TODO: update according to paper
        self.params.update({
            "n_subspaces":
            kwargs.get("n_subspaces", min(1000, int(self.shape[1]**2 / 2))),
            "subspace_size":
            kwargs.get("subspace_size", int(np.sqrt(self.shape[1]))),
            "n_jobs":
            kwargs.get("n_jobs", 1),
        })

    def _fit(self):
        subspaces = self._get_unique_subscapes()
        score_map = self._evaluate_subspaces(subspaces)
        importances = self._deduce_feature_importances(score_map)
        self.feature_importances.update(importances)

    def _get_unique_subscapes(self):
        """
        Return unique feature subspaces
        """
        # TODO improve to make sure to get the right amount of subspaces
        # sometimes lower because duplicated
        size = self.params["subspace_size"]
        if isinstance(size, tuple):
            lower, upper = size
        else:
            lower, upper = 1, size

        subspaces = [None] * self.params["n_subspaces"]
        for i in range(self.params["n_subspaces"]):
            m = np.random.randint(lower, upper + 1, 1)[0]
            f = list(np.random.choice(self.names, m, replace=False))
            subspaces[i] = sorted(f)

        subspaces.sort()
        return list(subspaces for subspaces, _ in itertools.groupby(subspaces))

    def _evaluate(self, subspace):
        features, types = self.data.get_subspace(subspace)
        score = self._evaluate_subspace(features, types)
        return {"features": subspace, "score": score}

    def _get_chunks(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _evaluate_subspaces(self, subspaces):
        """
        Collect and return subset evaluations

        Arguments:
            subspaces {list} -- List of feature subspaces
        """
        n_jobs = self.params["n_jobs"]
        if n_jobs == 1:
            return [self._evaluate(subspace) for subspace in subspaces]

        with Pool(n_jobs) as p:
            return p.map(self._evaluate, subspaces)
