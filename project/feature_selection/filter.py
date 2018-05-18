"""
    Mutual Information Transformer class
"""
from project.base import Selector
from project.shared import evaluate_subspace


class Filter(Selector):
    def __init__(self, f_types, l_type, shape, **kwargs):
        """
        Mutual Information FS class

        Arguments:
            f_types {pd.Series} -- Series containing feature types
            l_type {str} -- Type of label
            shape {tuple} -- Tuple containing the shape of features
        """
        super().__init__(f_types, l_type, shape, **kwargs)

    def _init_parameters(self, parameters):
        """
        Initialize parameters

        Arguments:
            parameters {dict} -- Parameter dict
        """
        self.params = {
            "k": parameters.get("k", int(self.shape[1] / 2 + 1)),
            "nominal_distance": parameters.get("nominal_distance", 1),
            "n_neighbors": parameters.get("n_neighbors", 6),
            "method": parameters.get("method", "mi"),
        }

    def _fit(self):
        self.data.add_salt()

        for col in self.data.X:
            X, types = self.data.get_subspace(col)
            self.feature_importances[col] = evaluate_subspace(
                X, self.data.y, types, self.l_type, self.domain, **self.params)
