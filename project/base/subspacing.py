"""
    Base class for subspacing approaches.
"""
import itertools
from abc import abstractmethod
from project.base import Selector
import numpy as np
from joblib import Parallel, delayed
from time import time


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
        self.params.update({
            "n_subspaces":
            kwargs.get("n_subspaces", min(1000, int(self.shape[1]**2 / 2))),
            "subspace_size":
            kwargs.get("subspace_size", int(np.sqrt(self.shape[1])))
        })

    def _fit(self):
        subspaces = self._get_unique_subscapes()
        start = time()
        score_map = self._evaluate_subspaces(subspaces)
        print("Sampling", time() - start)
        self.feature_importances = self._deduce_feature_importances(score_map)

    def _get_unique_subscapes(self):
        """
        Return unique feature subspaces
        """
        # TODO improve to make sure to get the right amount of subspaces
        names = self.data.X.columns
        size = self.params["subspace_size"]
        if isinstance(size, tuple):
            lower, upper = size
        else:
            lower, upper = 1, size

        subspaces = [None] * self.params["n_subspaces"]
        for i in range(self.params["n_subspaces"]):
            m = np.random.randint(lower, upper + 1, 1)[0]
            f = list(np.random.choice(names, m, replace=False))
            subspaces[i] = sorted(f)

        subspaces.sort()
        return list(subspaces for subspaces, _ in itertools.groupby(subspaces))

    def _evaluate(self, subspaces):
        results = [None] * len(subspaces)
        for i, subspace in enumerate(subspaces):
            features, types = self.data.get_subspace(subspace)
            score = self._evaluate_subspace(features, types)
            results[i] = {"features": subspace, "score": score}
        return results

    def _get_chunks(self, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _evaluate_subspaces(self, subspaces):
        """
        Collect and return subset evaluations

        Arguments:
            subspaces {list} -- List of feature subspaces
        """
        n_jobs = 4
        chunk_size = int(np.ceil(len(subspaces) / n_jobs))
        chunks = self._get_chunks(subspaces, chunk_size)
        knowledgebase = Parallel(
            n_jobs=n_jobs,
            verbose=100,
            batch_size=1,
            mmap_mode="r",
        )(delayed(self._evaluate)(chunk) for chunk in chunks)
        return list(itertools.chain.from_iterable(knowledgebase))
