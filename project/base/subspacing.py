"""
    Base class for subspacing approaches.
"""
import itertools
import numpy as np
from abc import abstractmethod

from project.base import Selector
from joblib import Parallel, delayed


class Subspacing(Selector):
    @abstractmethod
    def _evaluate_subspace(self, subspace):
        """
        Evaluate a feature subspace and return results as dict

        Arguments:
            subspace {list(str)} -- List containing the selected feature names
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
        n_subspaces = min(1000, self.shape[1] * int(np.sqrt(self.shape[1])))
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
        self.score_map = score_map
        importances = self._deduce_feature_importances(score_map)
        self.feature_importances.update(importances)

    def _get_unique_subscapes(self):
        """
        Return unique feature subspaces
        """
        size = self.params["subspace_size"]
        lower, upper = size if isinstance(size, tuple) else (1, size)

        n_subspaces, start = self.params["n_subspaces"], 0
        subspaces = [None] * n_subspaces

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

    def _evaluate(self, subspaces):
        results = [None] * len(subspaces)
        for i, subspace in enumerate(subspaces):
            score = self._evaluate_subspace(subspace)
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
        n_jobs = self.params["n_jobs"]
        if n_jobs == 1:
            return self._evaluate(subspaces)

        chunk_size = int(np.ceil(len(subspaces) / n_jobs))
        chunks = self._get_chunks(subspaces, chunk_size)

        backend = "threading" if self.params["eval_method"] == "rar" else None
        knowledgebase = Parallel(
            n_jobs=n_jobs,
            mmap_mode="r",
            max_nbytes="5G",
            backend=backend,
        )(delayed(self._evaluate)(chunk) for chunk in chunks)
        return list(itertools.chain.from_iterable(knowledgebase))
