import sys
import os
import requests
import json
import pandas as pd
from project import DATA_PATH


class Downloader():

    def __init__(self, name, file_type="arff"):
        self.name = name
        self.file_type = file_type
        self.folder = os.path.join(DATA_PATH, file_type, name)

    def download_data(self):
        source_dict = {
            "csv": self._download_uci_data,
            "arff": self._download_openml_data,
        }
        return source_dict[self.file_type]()

    def _download_uci_data(self):
        # Try to download dataset from UCI (unfortunately their naming is not consistent)
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
        # Try to download data from openml
        info_url = 'https://www.openml.org/api/v1/json/data/list/data_name/' + self.name

        try:
            # Get dataset id using api
            info_res = json.loads(requests.get(info_url).content)
            d_id = info_res["data"]["dataset"][0]["did"]

            # Get the details from website which includes also download link in url attribute
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
        # Write data to folder
        info_path = os.path.join(self.folder, "info.json")
        with open(info_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)

        data_path = os.path.join(self.folder, "data.arff")
        self._write_text_to_file(data_path, data)

    def _write_text_to_file(self, path, text):
        with open(path, 'w') as outfile:
            for line in text.split("\n"):
                outfile.write(line)
                outfile.write("\n")

    def _print_download_error(self, url):
        # UCI is inconsistent in their naming and some datasets need to be loaded manually
        print("Could not download data from url:\n{:s}".format(url))
        print("\nPlease visit the README for further instructions")
        sys.exit("Download Error")
