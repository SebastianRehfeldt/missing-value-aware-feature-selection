"""
    PSO class for feature selection
"""
from pyswarm import pso

from project.base import Selector
from project.shared.evaluation import evaluate_subspace


class PSO(Selector):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        self.params["eval_method"] = kwargs.get("eval_method", "tree")
        self._set_pso_options(**kwargs)
        self.params.update(self.default_options)

    def _set_pso_options(self, **kwargs):
        self.default_options = {
            "omega": kwargs.get("omega", 0.729844),
            "phip": kwargs.get("phip", 1.49618),
            "phig": kwargs.get("phig", 1.49618),
            "swarmsize": kwargs.get("swarmsize", 50),
            "maxiter": kwargs.get("maxiter", 100),
            "debug": kwargs.get("debug", False),
            "minfunc": kwargs.get("minfunc", 1e-4)
        }
        # Own modification
        self.default_options.update({
            "swarmsize":
            min(50, max(20, self.shape[1])),
            "maxiter":
            min(100, max(20, self.shape[1])),
        })

    def objective(self, x):
        if (x <= 0.6).all():
            return 1

        subspace = self.data.X.columns[x > 0.6].tolist()
        X, types = self.data.get_subspace(subspace)
        score = evaluate_subspace(
            X,
            self.data.y,
            types,
            self.data.l_type,
            self.domain,
            **self.params,
        )

        return 1 - score

    def _fit(self):
        """
        Calculate feature importances using pso
        """
        lb = [0] * self.data.shape[1]
        ub = [1] * self.data.shape[1]
        x_opt, _ = pso(self.objective, lb, ub, **self.default_options)

        for i, col in enumerate(self.data.X):
            self.feature_importances[col] = x_opt[i]
