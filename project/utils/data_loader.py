"""
    This class loads loads data and metadata and returns pandas dataframes which are ready to use.
    It can handle csv and arff files and download some datasets from UCI, such as ionosphere.

    from project.utils import DataLoader
    data_loader = DataLoader()
    features, labels, types = data_loader.load_data("ionosphere", "arff)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from scipy.io import arff
from project.utils.downloader import Downloader
from project.utils.csv_helper import CSVHelper


class DataLoader():

    def __init__(self, ignored_attributes=[], na_values=[], target_index=-1, *args, **kwargs):
        """
        Class for loading data

        Keyword Arguments:
            ignored_attributes {list} -- Feature to ignore (default: {[]})
            na_values {list} -- Missing values representations (default: {[]})
            target_index {int} -- Index of target (default: {-1})
            header {int, optional} -- Row number(s) to use as the column names (default: None)
            drop_unknown_samples {boolean, optional} -- Drop samples with missing class label
            nominal_thresh {int, optional} -- Threshold for unique values in numeric features
            names {list[str], optional} -- Feature names
            types {list[str], optional} -- Feature types
        """
        self.ignored_attributes = ignored_attributes
        self.na_values = ["?", "na"]
        self.na_values.extend(na_values)
        self.kwargs = kwargs

        # Parameters are ignored when data.csv and meta_data.json exist
        # Same for feature types and feature names
        # Modify meta_data.json to apply changes
        self.target = -1 if target_index == "last" else target_index

    def load_data(self, name="ionosphere", file_type="arff"):
        """
        Load data from disk or UCI/openml

        Directly supported datasets from UCI: ["ionosphere"]
        Openml: ["ionosphere", "isolet", "madelon", "musk", "semeion"]

        Keyword Arguments:
            name {str} -- Dataset name (default: {"ionosphere"})
            file_type {str} -- File type (default: {"arff"})
        """
        self.name = name
        self.file_type = file_type
        self.downloader = Downloader(name, file_type)

        if not os.path.isdir(self.downloader.folder):
            self.downloader.download_data()
        return self._load_data()

    def _load_data(self):
        """
        Read data from disk and prepare for analysis
        """
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
        if self.kwargs.get("drop_unknown_samples") == True:
            self._remove_samples_with_unknown_class()
        return self.data, self.labels, self.types

    ####################################################################################
    ###########################           ARFF           ###############################
    ####################################################################################

    def _load_arff(self):
        """
        Load arff data
        """
        file_path = os.path.join(self.downloader.folder, "data.arff")
        if not os.path.exists(file_path):
            self._print_file_not_found(file_path)

        data, meta = arff.loadarff(file_path)
        data = pd.DataFrame(data, columns=meta.names())
        types = pd.Series(meta.types(), meta.names())

        self._read_arff_config()
        return data, types

    def _read_arff_config(self):
        """
        Read arff config and set target and ignored attributes
        """
        arff_config = os.path.join(self.downloader.folder, "info.json")
        with open(arff_config) as json_data:
            config = json.load(json_data)
            self.target = config["default_target_attribute"]
            self.ignored_attributes.append(config["row_id_attribute"])
            self.ignored_attributes.append(config["ignore_attribute"])

    ####################################################################################
    ###########################           CSV           ################################
    ####################################################################################

    def _load_csv(self):
        """
        Read csv data and meta data
        """
        data = self._read_csv_data()
        types, names = self._read_csv_meta_data(data)
        data.columns = names
        data = self._encode_nominal_features(data, types)
        return data, types

    def _read_csv_data(self):
        """
        Read csv data from data.csv
        """
        path = os.path.join(self.downloader.folder, "data.csv")
        if not os.path.exists(path):
            self._print_file_not_found(path)

        data = pd.read_csv(path, header=self.kwargs.get("header"),
                           na_values=self.na_values, sep=None)
        if not self.kwargs.get("header") is None:
            self.kwargs["names"] = data.columns
        return data

    def _read_csv_meta_data(self, data):
        """
        Read meta data from disk or create if not present

        Arguments:
            data {pd dataframe} -- Dataframe needed for meta data creation
        """
        # Create meta data if not existing
        path = os.path.join(self.downloader.folder, "meta_data.json")
        if not os.path.exists(path):
            helper = CSVHelper(path, data, self.target, self.kwargs)
            names, types = helper.create_meta_data()

        # Load meta data
        meta_data = json.load(open(path))
        names = meta_data["feature_names"]
        types = pd.Series(meta_data["feature_types"], names)
        self.target = meta_data["feature_names"][self.target]
        return types, names

    def _encode_nominal_features(self, data, types):
        """
        Encode nominal features to match encoding from arff data

        Arguments:
            data {pd dataframe} -- Dataframe containing the features
            types {pd series} -- Feature types
        """
        indices = (i for i in range(data.shape[1]) if types[i] == "nominal")
        for i in indices:
            data.iloc[:, i].fillna("?", inplace=True)
            data.iloc[:, i] = data.iloc[:, i].apply(str).str.encode("utf-8")
        return data

    ####################################################################################
    ##########################           HELPER           ##############################
    ####################################################################################

    def _remove_attributes(self):
        """
        Remove attributes which are in ignored list
        """
        for attribute in self.ignored_attributes:
            if attribute in self.data.columns.tolist():
                self.data.drop(attribute, axis=1, inplace=True)
                self.types.drop(attribute, inplace=True)

    def _remove_samples_with_unknown_class(self):
        """
        Remove samples when their class label is missing
        """
        # Create mask for missing labels
        nominal_mask = self.labels.isin([b"?", np.nan])
        numerical_mask = pd.isna(self.labels)
        mask = np.logical_or(nominal_mask, numerical_mask)

        # Remove samples from dataframe and labels series
        self.labels = self.labels.loc[~mask].reset_index(drop=True)
        self.data = self.data.loc[~mask, :].reset_index(drop=True)

    def _print_file_not_found(self, file_path):
        """
        Error handling when folder exists, but without data.csv

        Arguments:
            file_path {str} -- Path to file
        """
        print("Could not find file: {:s}".format(file_path))
        sys.exit("File not found error")
