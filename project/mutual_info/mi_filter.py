import pandas as pd
from project.shared.selector import Selector
from project.mutual_info.mutual_information import get_mutual_information


class MI_Filter(Selector):
    def __init__(self, data, **kwargs):
        super().__init__(data)

    def _init_parameters(self, parameters):
        self.params = {
            "k": parameters.get("k", 3),
            "nominal_distance": parameters.get("nominal_distance", 1),
        }

    def calculate_feature_importances(self):
        scores = {}
        for col in self.data.features:
            features = self.data.features[col].to_frame()
            types = pd.Series(self.data.f_types[col], [col])

            selected_data = self.data._replace(
                features=features, f_types=types, shape=features.shape)

            scores[col] = get_mutual_information(selected_data)
        return scores
