"""
    This class loads loads data and metadata and returns panda dataframes which are ready to use.
    It can handle csv and arff files and download some datasets from UCI, such as ionosphere.

    from project.utils import DataLoader
    data_loader = DataLoader("ionosphere")
    features, labels, types = data_loader.load_data()
"""

import sys
import shutil
import os
import json
import pandas as pd
import numpy as np
from scipy.io import arff
from project import DATA_PATH
from project.utils.downloader import Downloader


class DataLoader():
    def __init__(self, target_index=-1, nominal_thresh=10, feature_names=None, feature_types=None, header=None, na_values=[], drop_unknown_samples=True, ignored_attributes=[]):
        self.header = header
        self.should_drop_unknown_samples = drop_unknown_samples
        self.na_values = ["?", "na"]
        self.na_values.extend(na_values)
        self.ignored_attributes = ignored_attributes

        # Parameters are ignored when data.csv and meta_data.json exist
        # Modify meta_data.json to apply changes
        self.target = -1 if target_index == "last" else target_index
        self.nominal_thresh = nominal_thresh
        self.feature_names = feature_names
        self.feature_types = feature_types

    ####################################################################################
    ##########################           HELPER           ##############################
    ####################################################################################

    def _print_file_not_found(self, file_path):
        # Error handling when folder exists, but without data.csv
        print("Could not find file: {:s}".format(file_path))
        sys.exit("File not found error")

    def _feature_is_sparse_int(self, feature, thresh=0.0001):
        # Features with missing values are always float64 even though data are integers
        # As the dtype cant be used, the complete data of the feature is analyzed
        # The float vector is converted to int (floor) and it is checked if the vectors dont differ significantly
        complete_vector = feature.dropna()
        distance = np.sum(complete_vector - complete_vector.astype(np.int))
        is_int = distance < (thresh * len(complete_vector))

        # We need to count unique values on complete vector as np.nan is always unique
        n_unique_values = len(np.unique(complete_vector))
        is_sparse = n_unique_values < self.nominal_thresh
        return is_sparse and is_int

    def _feature_is_nominal(self, feature):
        # A nominal feature is either an object or an int feature with few unique values
        if feature.dtype == "object":
            return True

        # A sparse integer is also considered as nominal feature
        return True if self._feature_is_sparse_int(feature) else False

    def _create_feature_names(self, n_features):
        # Create generic feature names and call target class
        self.feature_names = ["f{:d}".format(i) for i in range(n_features)]
        self.feature_names[self.target] = "class"

    def _create_feature_types(self, data, n_features):
        # Guess and set types of features
        self.feature_types = ["numeric"] * n_features
        self.feature_types[self.target] = "nominal"

        # Try to guess and update type for nominal features
        for i in range(n_features):
            feature = data.iloc[:, i]
            if self._feature_is_nominal(feature):
                self.feature_types[i] = "nominal"

        self.feature_types = pd.Series(self.feature_types, self.feature_names)

    def _store_meta_data(self, meta_path):
        # Write meta data to file
        meta_data = {
            "feature_names": self.feature_names,
            "feature_types": list(self.feature_types),
        }
        with open(meta_path, 'w') as outfile:
            json.dump(meta_data, outfile)

    def _create_meta_data(self, meta_path, data):
        # Compute meta data if it was not passed initially
        n_features = data.shape[1]

        if self.feature_names is None:
            self._create_feature_names(n_features)

        if self.feature_types is None:
            self._create_feature_types(data, n_features)

        self._store_meta_data(meta_path)

    def _encode_nominal_features(self, data, types):
        indices = (i for i in range(data.shape[1]) if types[i] == "nominal")
        for i in indices:
            data.iloc[:, i].fillna("?", inplace=True)
            data.iloc[:, i] = data.iloc[:, i].apply(str).str.encode("utf-8")
        return data

    def _remove_samples_with_unknown_class(self):
        nominal_mask = self.labels.isin([b"?", np.nan])
        numerical_mask = pd.isna(self.labels)
        mask = np.logical_or(nominal_mask, numerical_mask)

        self.labels = self.labels.loc[~mask].reset_index(drop=True)
        self.data = self.data.loc[~mask, :].reset_index(drop=True)

    def _remove_attributes(self):
        for attribute in self.ignored_attributes:
            if attribute in self.data.columns.tolist():
                self.data.drop(attribute, axis=1, inplace=True)
                self.types.drop(attribute, inplace=True)

    ####################################################################################
    #########################           LOADING           ##############################
    ####################################################################################

    def _read_csv_data(self):
        # Read data from disk (data.csv need to exist in the right folder)
        path = os.path.join(self.downloader.folder, "data.csv")
        if not os.path.exists(path):
            self._print_file_not_found(path)

        return pd.read_csv(path, header=self.header, na_values=self.na_values, sep=None)

    def _read_csv_meta_data(self, data):
        # Read meta data from disk or create if not present
        file_path = os.path.join(self.downloader.folder, "meta_data.json")
        if not os.path.exists(file_path):
            self._create_meta_data(file_path, data)

        meta_data = json.load(open(file_path))
        names = meta_data["feature_names"]
        types = pd.Series(meta_data["feature_types"], names)
        self.target = meta_data["feature_names"][self.target]
        return types, names

    def _read_arff_config(self):
        arff_config = os.path.join(self.downloader.folder, "info.json")
        with open(arff_config) as json_data:
            config = json.load(json_data)
            self.target = config["default_target_attribute"]
            self.ignored_attributes.append(config["row_id_attribute"])
            self.ignored_attributes.append(config["ignore_attribute"])

    def _load_csv(self):
        # Read data and meta data and store them in class variables
        data = self._read_csv_data()
        types, names = self._read_csv_meta_data(data)
        data.columns = names
        data = self._encode_nominal_features(data, types)
        return data, types

    def _load_arff(self):
        file_path = os.path.join(self.downloader.folder, "data.arff")
        if not os.path.exists(file_path):
            self._print_file_not_found(file_path)

        data, meta = arff.loadarff(file_path)
        data = pd.DataFrame(data, columns=meta.names())
        types = pd.Series(meta.types(), meta.names())

        self._read_arff_config()
        return data, types

    def _load_data(self):
        load_dict = {
            "arff": self._load_arff,
            "csv": self._load_csv,
        }
        # Read data and remove ignored attributes
        self.data, self.types = load_dict[self.file_type]()
        self._remove_attributes()

        # Split features and labels
        self.labels = self.data[self.target]
        self.data = self.data.drop(self.target, axis=1)

        # Remove samples with missing class information
        if self.should_drop_unknown_samples:
            self._remove_samples_with_unknown_class()

        return self.data, self.labels, self.types

    def load_data(self, name, file_type="arff"):
        self.name = name
        self.file_type = file_type
        self.downloader = Downloader(name, file_type)

        if not os.path.isdir(self.downloader.folder):
            self.downloader.download_data()

        return self._load_data()
