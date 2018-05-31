"""
    Base class for subspacing approaches.
"""
from abc import abstractmethod
from multiprocessing import Pool
import numpy as np

from project.base import Selector


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

        # as suggested by rknn
        n_subspaces = max(1000, int(np.sqrt(self.shape[1]**2 / 2)))
        n_subspaces = kwargs.get("n_subspaces", n_subspaces)

        size = int(np.sqrt(self.shape[1]))
        subspace_size = kwargs.get("subspace_size", size)

        self.params.update({
            "n_subspaces": n_subspaces,
            "subspace_size": subspace_size,
            "n_jobs": kwargs.get("n_jobs", 1),
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
        size = self.params["subspace_size"]
        lower, upper = size if isinstance(size, tuple) else (1, size)

        # evaluate each feature independently
        n_subspaces, start = self.params["n_subspaces"], 0
        subspaces = [None] * n_subspaces
        if lower == 1:
            for i, name in enumerate(self.names):
                subspaces[i] = [name]
            start, lower = len(self.names), 2

        # add multi-d subspaces
        max_retries = 10
        for i in range(start, n_subspaces):
            found_new, retries = False, 0
            while not found_new and retries < max_retries:
                m = np.random.randint(lower, upper + 1, 1)[0]
                f = sorted(np.random.choice(self.names, m, replace=False))
                found_new = f not in subspaces
                retries += 1
            subspaces[i] = f
        return subspaces

    def _evaluate(self, subspace):
        features, types = self.data.get_subspace(subspace)
        score = self._evaluate_subspace(features, types)
        return {"features": subspace, "score": score}

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
