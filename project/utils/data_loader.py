#%%
import os
import requests
import pandas as pd
from project import DATA_PATH


def _load_uci_data(folder):
    file_path = os.path.join(folder, "data.csv")
    df = pd.read_csv(file_path, header=None)
    return df


def _download_info(folder, url):
    try:
        output_path = os.path.join(folder, "names.txt")
        text = requests.get(url).text
        with open(output_path, 'w') as f:
            for line in text.split("\n"):
                f.write(line)
                f.write("\n")
    except:
        print("could not download data from url:\n{:s}".format(url))
        raise

def _download_data(folder, url):
    try:
        output_path = os.path.join(folder, "data.csv")
        data = pd.read_csv(url)
        data.to_csv(output_path, sep=',', encoding='utf-8', index=False)
    except:
        print("could not download data from url:\n{:s}".format(url))
        raise

def _download_data_from_uci(folder, name):
    os.makedirs(folder)

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    base_url = "{:s}/{:s}/{:s}.".format(base_url, name, name)

    _download_info(folder, base_url + "names")
    _download_data(folder, base_url + "data")


def load_data_from_uci(name="ionosphere"):
    folder = os.path.join(DATA_PATH, name)
    data_already_downloaded = os.path.isdir(folder)
    if not data_already_downloaded:
        return _download_data_from_uci(folder, name)
    return _load_uci_data(folder)
