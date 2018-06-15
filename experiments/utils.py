import os
import json


def write_config(folder, config, dataset_config, algorithms):
    filename = os.path.join(folder, "config.txt")
    with open(filename, 'w+') as file:
        file.write("CONFIG EXPERIMENT\n\n")
        file.write(json.dumps(config, indent=4))
        file.write("\n\nCONFIG DATASET\n\n")
        file.write(json.dumps(dataset_config, indent=4))
        file.write("\n\nCONFIG ALGORITHMS")

        for key, algorithm in algorithms.items():
            shape = (dataset_config["n_samples"], dataset_config["n_features"])
            data = [[], "", shape]
            selector = algorithm["class"](*data, **algorithm["config"])
            params = selector.get_params()
            del params["f_types"]
            del params["l_type"]
            del params["shape"]
            file.write("\n\nCONFIG - {:s}\n".format(key))
            file.write(json.dumps(params, indent=4))
