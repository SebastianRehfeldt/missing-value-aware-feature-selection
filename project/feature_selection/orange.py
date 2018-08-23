import warnings
from project.base import Selector
from Orange.preprocess.score import ReliefF, FCBF
from Orange.classification import RandomForestLearner


class Orange(Selector):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        self.params["eval_method"] = kwargs.get("eval_method", "relief")

    def _fit(self):
        warnings.simplefilter(action='ignore')
        table = self.data.to_table()
        if self.params["eval_method"] == "relief":
            scores = ReliefF(table, n_iterations=100)
        elif self.params["eval_method"] == "fcbf":
            scores = FCBF(table)
        else:
            scores = RandomForestLearner().score_data(table)[0]

        for attr, score in zip(table.domain.attributes, scores):
            self.feature_importances[attr.name] = score
