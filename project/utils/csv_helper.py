"""
    Helper class for creating meta data from csv file.
    Trys to guess type for feature if not given. 
"""
import json
import pandas as pd
import numpy as np


class CSVHelper():

    def __init__(self, path, data, target, *args, **kwargs):
        """
        Class for creating csv met data

        Arguments:
            path {str} -- Path to file
            data {pd dataframe} -- Dataframe with features and labels
            target {int} -- Index of target

        Keyword Arguments:
            nominal_thresh {int, optional} -- Threshold for unique values in numeric features
            names {list[str], optional} -- Feature names
            types {list[str], optional} -- Feature types
        """
        self.path = path
        self.data = data
        self.n_features = data.shape[1]
        self.target = target
        self.kwargs = kwargs

    def create_meta_data(self):
        """
        Create and store meta data
        """
        self.names = self.kwargs.get("names") or self._create_feature_names()
        self.types = self.kwargs.get("types") or self._create_feature_types()
        self._store_meta_data()
        return self.names, self.types

    def _create_feature_names(self):
        """
        Creates and returns list of generic feature names
        """
        self.names = ["f{:d}".format(i) for i in range(self.n_features)]
        self.names[self.target] = "class"
        return self.names

    def _create_feature_types(self):
        """
        Creates and returns list of feature types
        """
        self.types = ["numeric"] * self.n_features
        self.types[self.target] = "nominal"

        # Try to guess and update type for nominal features
        for i in range(self.n_features):
            feature = self.data.iloc[:, i]
            if self._feature_is_nominal(feature):
                self.types[i] = "nominal"

        self.types = pd.Series(self.types, self.names)
        return self.types

    def _feature_is_nominal(self, feature):
        """
        Checks if feature is nominal

        Arguments:
            feature {pd dataframe} -- Column of dataframe
        """
        if feature.dtype == "object":
            return True

        # A sparse integer is also considered as nominal feature
        return True if self._feature_is_sparse_int(feature) else False

    def _feature_is_sparse_int(self, feature, thresh=0.0001):
        """
        Checks if feature is a sparce integer

        Arguments:
            feature {pd dataframe} -- Column of dataframe

        Keyword Arguments:
            thresh {float} -- Treshold to tackle possible precision errors (default: {0.0001})
        """
        # Features with missing values are always float64 even though data are integers
        # As the dtype cant be used, the complete data of the feature is analyzed
        # The float vector is converted to int (floor) and it is checked if the vectors dont differ significantly
        complete_vector = feature.dropna()
        distance = np.sum(complete_vector - complete_vector.astype(np.int))
        is_int = distance < (thresh * len(complete_vector))

        # We need to count unique values on complete vector as np.nan is always unique
        n_unique_values = len(np.unique(complete_vector))
        is_sparse = n_unique_values < (self.kwargs.get("nominal_thresh") or 10)
        return is_sparse and is_int

    def _store_meta_data(self):
        """
        Stores meta data at file
        """
        meta_data = {
            "feature_names": self.names,
            "feature_types": list(self.types),
        }
        with open(self.path, 'w') as outfile:
            json.dump(meta_data, outfile)
