#%%
"""
    This class loads loads data and metadata and returns panda dataframes which are ready to use.
    It can handle csv and arff files and download some datasets from UCI, such as ionosphere.

    from project.utils import DataLoader
    data_loader = DataLoader("ionosphere")
    features, labels, types = data_loader.load_csv()
"""

# Author: Sebastian Rehfeldt <sebastian.rehfeldt@gmx.net

import sys
import shutil
import os
import requests
import json
import pandas as pd
import numpy as np
from scipy.io import arff
from project import DATA_PATH

class DataLoader():
    def __init__(self, name, target_index=-1, nominal_thresh=10, feature_names=None, feature_types=None, header=None):
        self.name           = name
        self.csv_folder     = os.path.join(DATA_PATH, "csv", name)
        self.arff_folder    = os.path.join(DATA_PATH, "arff", name)
        self.header         = header

        # Parameters are ignored when data.csv and meta_data.json exist
        # Modify meta_data.json to apply changes
        self.target         = -1 if target_index == "last" else target_index
        self.nominal_thresh = nominal_thresh
        self.feature_names  = feature_names
        self.feature_types  = feature_types
            
    
    ####################################################################################
    #########################           DOWNLOAD           #############################
    ####################################################################################
    def _download_data_from_uci(self):
        # Download data from UCI into newly created folder
        os.makedirs(self.csv_folder)

        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
        base_url = "{:s}/{:s}/{:s}.".format(base_url, self.name, self.name)

        self._download_info(base_url + "names")
        self._download_data(base_url + "data")
    
    def _download_info(self, url):
        # Try to download dataset information
        try:
            output_path = os.path.join(self.csv_folder, "names.txt")
            text = requests.get(url).text
            with open(output_path, 'w') as outfile:
                for line in text.split("\n"):
                    outfile.write(line)
                    outfile.write("\n")
        except:
            self._print_download_error(url)

    def _download_data(self, url):
        # Try to download dataset features
        try:
            data = pd.read_csv(url)
            data_path = os.path.join(self.csv_folder, "data.csv")
            data.to_csv(data_path, sep=',', encoding='utf-8', index=False)
        except:
            self._print_download_error(url)

    def _print_download_error(self, url):
        # UCI is inconsistent in their naming and some datasets need to be loaded manually
        print("Could not download data from url:\n{:s}".format(url))
        print("\nPlease visit the README for further instructions")
        shutil.rmtree(self.csv_folder)
        sys.exit("Download Error")

    def _download_openml_data(self):
        try:
            info_url = 'https://www.openml.org/api/v1/json/data/list/data_name/' + self.name
            info_res = json.loads(requests.get(info_url).content)
            d_id     = info_res["data"]["dataset"][0]["did"]

            details_url     = "https://www.openml.org/d/{:d}/json".format(d_id)
            dataset_infos   = json.loads(requests.get(details_url).content)
            arff_file       = requests.get(dataset_infos["url"]).content

            os.makedirs(self.arff_folder)
            self._write_openml_data(dataset_infos, arff_file)
        except:
            print("Dataset could not be located on disk or on openml")
            sys.exit("Could not load: {:s}".format(self.name))
        
    def _write_openml_data(self, dataset_infos, arff_file):
        info_path = os.path.join(self.arff_folder, "{:s}.json".format(self.name))
        with open(info_path, 'w') as outfile:
            json.dump(dataset_infos, outfile, indent=4)

        data_path = os.path.join(self.arff_folder, "{:s}.arff".format(self.name))
        with open(data_path, 'w') as outfile:
            for line in arff_file.decode().split("\n"):
                outfile.write(line)
                outfile.write("\n")          

    ####################################################################################
    ##########################           HELPER           ##############################
    ####################################################################################
    def _print_file_not_found(self, file_path):
        # Error handling when folder exists, but without data.csv
        print("Could not find file: {:s}".format(file_path))
        sys.exit("File not found error")

    def _feature_is_nominal(self, feature):
        # A nominal feature is either an object or an int feature with few unique values
        unique_features = len(np.unique(feature))
        is_object       = feature.dtype == "object"
        is_sparse_int   = feature.dtype == "int64" and unique_features < self.nominal_thresh
        return True if is_object or is_sparse_int else False

    def _create_feature_names(self, n_features):
        # Create generic feature names and call target class
        self.feature_names = ["f{:d}".format(i) for i in range(n_features)]
        self.feature_names[self.target] = "class"

    def _create_feature_types(self, n_features):
        # Guess and set types of features
        self.feature_types = ["numeric"] * n_features
        self.feature_types[self.target] = "nominal"
        
        # Try to guess and update type for nominal features
        for i in range(n_features):
            feature = self.data.iloc[:,i]
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

    def _create_meta_data(self, meta_path):
        # Compute meta data if it was not passed initially
        n_features = self.data.shape[1]

        if self.feature_names is None:
            self._create_feature_names(n_features)
        
        if self.feature_types is None:
            self._create_feature_types(n_features)

        self._store_meta_data(meta_path)


    ####################################################################################
    #########################           LOADING           ##############################
    ####################################################################################
    def _read_csv(self):
        # Read data from disk (data.csv need to exist in the right folder)
        file_path = os.path.join(self.csv_folder, "data.csv")
        if not os.path.exists(file_path):
            self._print_file_not_found(file_path)
        self.data = pd.read_csv(file_path, header=self.header)

    def _read_meta_data(self):
        # Read meta data from disk or create if not present
        meta_path = os.path.join(self.csv_folder, "meta_data.json")
        if not os.path.exists(meta_path):
            self._create_meta_data(meta_path)
        else:
            meta_data = json.load(open(meta_path))
            self.feature_names = meta_data["feature_names"]
            self.feature_types = pd.Series(meta_data["feature_types"], self.feature_names)

    def _load_csv(self):
        # Read data and meta data and store them in class variables 
        self._read_csv()
        self._read_meta_data()

        # Split data and labels
        self.data.columns   = self.feature_names
        self.labels         = self.data["class"]
        self.data           = self.data.drop("class", axis=1)
        return self.data, self.labels, self.feature_types

    def _load_arff(self):
        arff_file = os.path.join(self.arff_folder, "{:s}.arff".format(self.name))
        data, meta = arff.loadarff(arff_file)
        
        self.feature_names  = [name.lower() for name in meta.names()]
        self.feature_types  = pd.Series(meta.types(), self.feature_names)
        self.data           = pd.DataFrame(data, columns=self.feature_names)
        
        if "id" in self.feature_names:
            self.feature_names.remove("id")
            self.data           = self.data.drop("id", axis=1)
            self.feature_types  = self.feature_types.drop("id")

        self.labels         = self.data["class"]
        self.data           = self.data.drop("class", axis=1)

        return self.data, self.labels, self.feature_types

    def load_csv(self):
        # Try to download data from UCI if data not exists locally
        data_already_downloaded = os.path.isdir(self.csv_folder)
        if not data_already_downloaded:
            self._download_data_from_uci()

        return self._load_csv()

    def load_arff(self):
        data_already_downloaded = os.path.isdir(self.arff_folder)
        if not data_already_downloaded:
            self._download_openml_data()

        return self._load_arff()


    def load_data(self, file_type="arff"):
        return self.load_arff() if file_type=="arff" else self.load_csv()

data_loader = DataLoader("musk")
features, labels, feature_types = data_loader.load_data()
print(features)
