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
from project.utils.csv_helper import CSVHelper


class DataLoader():
    def __init__(self, header=None, drop_unknown_samples=True, ignored_attributes=[], na_values=[], target_index=-1, nominal_thresh=10, feature_names=None, feature_types=None):
        self.header = header
        self.should_drop_unknown_samples = drop_unknown_samples
        self.ignored_attributes = ignored_attributes
        self.na_values = ["?", "na"]
        self.na_values.extend(na_values)

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

    def _encode_nominal_features(self, data, types):
        indices = (i for i in range(data.shape[1]) if types[i] == "nominal")
        for i in indices:
            data.iloc[:, i].fillna("?", inplace=True)
            data.iloc[:, i] = data.iloc[:, i].apply(str).str.encode("utf-8")
        return data

    def _remove_attributes(self):
        for attribute in self.ignored_attributes:
            if attribute in self.data.columns.tolist():
                self.data.drop(attribute, axis=1, inplace=True)
                self.types.drop(attribute, inplace=True)

    def _remove_samples_with_unknown_class(self):
        nominal_mask = self.labels.isin([b"?", np.nan])
        numerical_mask = pd.isna(self.labels)
        mask = np.logical_or(nominal_mask, numerical_mask)

        self.labels = self.labels.loc[~mask].reset_index(drop=True)
        self.data = self.data.loc[~mask, :].reset_index(drop=True)

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
        path = os.path.join(self.downloader.folder, "meta_data.json")
        if not os.path.exists(path):
            helper = CSVHelper(path, data, self.target, self.feature_names,
                               self.feature_types, self.nominal_thresh)
            self.feature_names, self.feature_types = helper.create_meta_data()

        meta_data = json.load(open(path))
        names = meta_data["feature_names"]
        types = pd.Series(meta_data["feature_types"], names)
        self.target = meta_data["feature_names"][self.target]
        return types, names

    def _load_csv(self):
        # Read data and meta data and store them in class variables
        data = self._read_csv_data()
        types, names = self._read_csv_meta_data(data)
        data.columns = names
        data = self._encode_nominal_features(data, types)
        return data, types

    def _read_arff_config(self):
        arff_config = os.path.join(self.downloader.folder, "info.json")
        with open(arff_config) as json_data:
            config = json.load(json_data)
            self.target = config["default_target_attribute"]
            self.ignored_attributes.append(config["row_id_attribute"])
            self.ignored_attributes.append(config["ignore_attribute"])

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
        # Read data and remove ignored attributes
        self.data, self.types = {
            "arff": self._load_arff,
            "csv": self._load_csv,
        }[self.file_type]()
        self._remove_attributes()

        # Split features and labels
        self.labels = self.data[self.target]
        self.data = self.data.drop(self.target, axis=1)

        # Remove samples with missing class information
        if self.should_drop_unknown_samples:
            self._remove_samples_with_unknown_class()
        return self.data, self.labels, self.types

    def load_data(self, name="ionosphere", file_type="arff"):
        self.name = name
        self.file_type = file_type
        self.downloader = Downloader(name, file_type)

        if not os.path.isdir(self.downloader.folder):
            self.downloader.download_data()
        return self._load_data()
