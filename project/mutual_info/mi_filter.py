"""
    Mutual Information Transformer class
"""
from project.shared.selector import Selector
from project.mutual_info.mutual_information import get_mutual_information


class MI_Filter(Selector):
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
        }

    def _fit(self):
        self.data.add_salt()

        for col in self.data.X:
            X, types = self.data.get_subspace(col)
            self.feature_importances[col] = get_mutual_information(
                X, self.data.y, types, self.data.l_type,
                self.params["n_neighbors"])
