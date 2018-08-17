"""
    Base class for subspacing approaches.
"""
import itertools
import numpy as np
from abc import abstractmethod
from joblib import Parallel, delayed

from project.base import Selector


class Subspacing(Selector):
    @abstractmethod
    def _evaluate_subspace(self, subspace):
        raise NotImplementedError(
            "subclasses must implement _evaluate_subspace")

    @abstractmethod
    def _deduce_feature_importances(self, score_map):
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
        self.score_map = self._evaluate_subspaces(subspaces)
        scores = self._deduce_feature_importances(self.score_map)
        self.feature_importances.update(scores)

    def _get_p(self):
        p = None
        if self.params.get("active_sampling", False):
            p = np.ones(self.data.shape[1])

            if self.params.get("active_sampling_mr", False):
                p += self.missing_rates.values * 2

            if self.params.get("active_sampling_corr", False):
                p += (1 - self.nan_correlation.mean().values) * 5

            p /= np.sum(p)
        return p

    def _get_next(self, dim, p, subspaces, upper):
        found_new, retries = False, 0
        while not found_new and retries < 3:
            f = sorted(np.random.choice(self.names, dim, False, p))
            found_new = f not in subspaces
            retries += 1
            if dim == 1 and retries > 2:
                dim = np.random.randint(2, upper + 1, 1)[0]
        return f

    def _get_unique_subscapes(self):
        size = self.params["subspace_size"]
        lower, upper = size if isinstance(size, tuple) else (1, size)

        n_subspaces = self.params["n_subspaces"]
        subspaces = [None] * n_subspaces

        p = self._get_p()
        dims = np.random.randint(lower, upper + 1, n_subspaces)
        for i in range(n_subspaces):
            subspaces[i] = self._get_next(dims[i], p, subspaces, upper)

        subspaces.sort(key=len)
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
