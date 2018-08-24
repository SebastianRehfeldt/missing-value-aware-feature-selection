from project.base import Selector


class Baseline(Selector):
    def _fit(self):
        for i, col in enumerate(self.data.X):
            self.feature_importances[col] = 1 / (i + 2)
