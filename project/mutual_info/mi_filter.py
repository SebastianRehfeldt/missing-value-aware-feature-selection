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
        }

    def calculate_feature_importances(self):
        """
        Calculate importances for each single feature
        """
        scores = {}
        for col in self.data.X:
            X, types = self.data.get_subspace(col)
            scores[col] = get_mutual_information(X, self.data.y, types,
                                                 self.data.l_type)
        return scores
