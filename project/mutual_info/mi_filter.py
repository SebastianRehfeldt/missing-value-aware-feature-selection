"""
    Mutual Information Transformer class 
"""
import pandas as pd
from project.shared.selector import Selector
from project.mutual_info.mutual_information import get_mutual_information
from project.utils.assertions import assert_df, assert_types


class MI_Filter(Selector):
    def __init__(self, data, **kwargs):
        """
        Mutual Information FS class 

        Arguments:
            data {data} -- Data object which is used for FS
        """
        super().__init__(data)

    def _init_parameters(self, parameters):
        """
        Initialize parameters

        Arguments:
            parameters {dict} -- Parameter dict
        """
        self.params = {
            "k": parameters.get("k", 3),
            "nominal_distance": parameters.get("nominal_distance", 1),
        }

    def calculate_feature_importances(self):
        """
        Calculate importances for each single feature
        """
        scores = {}
        for col in self.data.X:
            # TODO test for mutli-d calls
            new_data = self.data.select(col)
            scores[col] = get_mutual_information(new_data)
        return scores
