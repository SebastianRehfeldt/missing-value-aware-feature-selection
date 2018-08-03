from project.base import Selector
from project.classifier import SKClassifier


class Embedded(Selector):
    def _fit(self):
        classifier = SKClassifier(self.data.f_types, kind="xgb")
        classifier.fit(self.data.X, self.data.y)

        scores = classifier.clf.feature_importances_
        for i in range(len(scores)):
            name = self.names[i]
            self.feature_importances[name] = scores[i]
