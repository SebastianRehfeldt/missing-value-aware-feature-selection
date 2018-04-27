#%%
import sys
import shutil
import os
import requests
import json
import pandas as pd
import numpy as np
from project import DATA_PATH


def _load_uci_data(folder, target, nominal_thresh):
    file_path = os.path.join(folder, "data.csv")
    data = pd.read_csv(file_path, header=None)
    meta_path = os.path.join(folder, "meta_data.json")
    if not os.path.exists(meta_path):
        meta_data = _get_meta_data(data, target, nominal_thresh)
        with open(meta_path, 'w') as outfile:
            json.dump(meta_data, outfile)
    else:
        meta_data = json.load(open(meta_path))

    data.columns = meta_data["feature_names"]
    index_of_class = data.columns.get_loc("class")
    labels = data["class"]
    data = data.drop("class", axis=1)
    feature_types = meta_data["feature_types"]
    del feature_types[index_of_class]
    return data, labels, np.asarray(feature_types)
        

def feature_is_nominal(feature, nominal_thresh):
    is_object       = feature.dtype == "object"
    is_sparse_int   = feature.dtype == "int64" and len(np.unique(feature)) < nominal_thresh
    return True if is_object or is_sparse_int else False


def _get_meta_data(data, target, nominal_thresh):
    amount_of_features = data.shape[1]

    types = ["numeric"] * amount_of_features
    names = ["f{:d}".format(i) for i in range(amount_of_features)]

    target = -1 if target=="last" else target
    types[target], names[target] = "nominal", "class"

    for i in range(amount_of_features):
        feature = data.iloc[:,i]
        if feature_is_nominal(feature, nominal_thresh):
            types[i] = "nominal"

    return { 
        "feature_names": names,
        "feature_types": types,
    }


def _print_download_error(folder, url):
    print("Could not download data from url:\n{:s}".format(url))
    print("\nPlease visit the README for further instructions")
    shutil.rmtree(folder)
    sys.exit("Download Error")


def _download_info(folder, url):
    try:
        output_path = os.path.join(folder, "names.txt")
        text = requests.get(url).text
        with open(output_path, 'w') as f:
            for line in text.split("\n"):
                f.write(line)
                f.write("\n")
    except:
        _print_download_error(folder, url)


def _download_data(folder, url, target, nominal_thresh):
    try:
        data = pd.read_csv(url)
        data_path = os.path.join(folder, "data.csv")
        data.to_csv(data_path, sep=',', encoding='utf-8', index=False)
    except:
        _print_download_error(folder, url)
        

def _download_data_from_uci(folder, name, target, nominal_thresh):
    os.makedirs(folder)

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    base_url = "{:s}/{:s}/{:s}.".format(base_url, name, name)

    _download_info(folder, base_url + "names")
    _download_data(folder, base_url + "data", target, nominal_thresh)


def load_data_from_uci(name="ionosphere", target=-1, nominal_thresh=10):
    folder = os.path.join(DATA_PATH, name)
    data_already_downloaded = os.path.isdir(folder)
    if not data_already_downloaded:
        _download_data_from_uci(folder, name, target, nominal_thresh)
    return _load_uci_data(folder, target, nominal_thresh)
