"""
    Utils class for scaling the data to achieve zero mean and unit deviation
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from project.utils.assertions import assert_data


def scale_data(data, method="standard"):
    """
    Scale data using standard scaler

    Arguments:
        data {Data} -- Data object which should be scaled

    Keyword Arguments:
        method {str} -- [description] (default: {"standard"})
    """
    scaler = {
        "standard": StandardScaler,
    }.get(method, StandardScaler)()

    new_features = data.features
    for col in data.features:
        # Do not scale nominal features
        if data.f_types[col] == "nominal":
            continue

        # Fit scaler on complete data
        feature = data.features[col]
        complete_cases = feature.dropna()
        scaler.fit(complete_cases.reshape(-1, 1))

        # Get indices of missing values, fill with zeros, scale
        missing_indices = feature.isnull()
        feature.fillna(0, inplace=True)
        feature = scaler.transform(feature.reshape(-1, 1))

        # Set missing values nan again and update dataframe
        feature[missing_indices] = np.nan
        new_features[col] = feature

    new_data = data._replace(features=new_features)
    new_data = assert_data(new_data)
    return new_data
