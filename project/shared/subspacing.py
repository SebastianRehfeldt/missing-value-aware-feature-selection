import itertools
from abc import ABC, abstractmethod
import numpy as np


class Subspacing(ABC):
    def __init__(self, data, **kwargs):
        self.data = data
        self.is_fitted = False
        self._init_parameters(kwargs)

    @abstractmethod
    def _init_parameters(self, parameters):
        self.params = {}
        raise NotImplementedError("subclasses must override _init_parameters")

    @abstractmethod
    def _evaluate_subspace(self, features):
        raise NotImplementedError(
            "subclasses must implement _evaluate_subspace")

    @abstractmethod
    def _deduce_feature_importances(self, score_map):
        raise NotImplementedError(
            "subclasses must implement _deduce_feature_importances")

    def _get_unique_subscapes(self):
        # TODO allow a range for m
        subspaces = [list(np.random.choice(self.data.features.columns, self.params["m"], replace=False))
                     for i in range(self.params["n"])]
        subspaces.sort()
        return list(subspaces for subspaces, _ in itertools.groupby(subspaces))

    def _evaluate_subspaces(self, subspaces):
        knowledgebase = []
        for subspace in subspaces:
            # TODO asssert features is dataframe
            features = self.data.features[subspace]
            knowledgebase.append({
                "features": subspace,
                "score": self._evaluate_subspace(features)
            })
        return knowledgebase
