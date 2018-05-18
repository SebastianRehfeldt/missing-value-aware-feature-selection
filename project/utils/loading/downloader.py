"""
    This class is responsible for downloading data from UCI ML or openml.
    The first is used for csv and the second for arff which are recommended.
"""
import sys
import os
import requests
import json
import pandas as pd
from project import DATA_PATH


class Downloader():
    def __init__(self, name, file_type="arff"):
        """
        Class for obtaining data from UCI or openml.

        Arguments:
            name {str} -- Name of the dataset.

        Keyword Arguments:
            file_type {str} -- Specify if "arff" or "csv" (default: {"arff"})
        """
        self.name = name
        self.file_type = file_type
        self.folder = os.path.join(DATA_PATH, file_type, name)

    def download_data(self):
        """
        Downloads data from uci ("csv") or openml ("arff")
        """
        return {
            "csv": self._download_uci_data,
            "arff": self._download_openml_data,
        }[self.file_type]()

    def _download_uci_data(self):
        """
        Downloads data from UCI and stores itin data folder.
        """
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
        base_url = "{:s}/{:s}/{:s}.".format(base_url, self.name, self.name)

        try:
            # Download info
            output_path = os.path.join(self.folder, "names.txt")
            info = requests.get(base_url + "names").text

            # Download data
            data = pd.read_csv(base_url + "data")
            data_path = os.path.join(self.folder, "data.csv")

            # Create folder and store the data there
            os.makedirs(self.folder)
            self._write_text_to_file(output_path, info)
            data.to_csv(data_path, sep=',', encoding='utf-8', index=False)
        except:
            self._print_download_error(base_url)

    def _download_openml_data(self):
        """
        Downloads data from UCI and stores it in data folder.
        """
        info_url = 'https://www.openml.org/api/v1/json/data/list/data_name/'
        info_url += self.name

        try:
            # Get dataset id using api
            info_res = json.loads(requests.get(info_url).content)
            d_id = info_res["data"]["dataset"][0]["did"]

            # Get the details from website
            details_url = "https://www.openml.org/d/{:d}/json".format(d_id)
            dataset_infos = json.loads(requests.get(details_url).content)

            # Download and decode the arff file
            arff_file = requests.get(dataset_infos["url"]).content.decode()

            # Create folder and store data there
            os.makedirs(self.folder)
            self._write_openml_data_to_files(dataset_infos, arff_file)
        except:
            self._print_download_error(info_url)

    def _write_openml_data_to_files(self, info, data):
        """
        Write openml info and data into files at disk

        Arguments:
            info {dict} -- Dict containing all dataset information
            data {src} -- arff file encoded as string
        """
        info_path = os.path.join(self.folder, "info.json")
        with open(info_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)

        data_path = os.path.join(self.folder, "data.arff")
        self._write_text_to_file(data_path, data)

    def _write_text_to_file(self, path, text):
        """
        Helper to write text to disk while keeping linebreaks

        Arguments:
            path {str} -- Path to file on disk
            text {str} -- Text which should be written to disk
        """
        with open(path, 'w') as outfile:
            for line in text.split("\n"):
                outfile.write(line)
                outfile.write("\n")

    def _print_download_error(self, url):
        """
        UCI has no api and some datasets need to be loaded manually

        Arguments:
            url {str} -- Url to the file at UCI which is not existing
        """
        print("Could not download data from url:\n{:s}".format(url))
        print("\nPlease visit the README for further instructions")
        sys.exit("Download Error")
