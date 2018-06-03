"""
    Mutual Information Transformer class
"""
from project.base import Selector
from project.shared.evaluation import evaluate_subspace


class Filter(Selector):
    def _fit(self):
        for col in self.data.X:
            X, types = self.data.get_subspace(col)
            self.feature_importances[col] = evaluate_subspace(
                X, self.data.y, types, self.l_type, self.domain, **self.params)
