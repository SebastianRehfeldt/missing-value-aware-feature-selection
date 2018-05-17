"""
    Utils class for scaling the data to achieve zero mean and unit deviation
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


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

    new_features = data.X
    for col in data.X:
        # Do not scale nominal features
        if data.f_types[col] == "nominal":
            continue

        # Fit scaler on complete data
        feature = data.X[col]
        complete_cases = feature.dropna()
        scaler.fit(complete_cases.values.reshape(-1, 1))

        # Get indices of missing values, fill with zeros, scale
        missing_indices = feature.isnull()
        feature.fillna(0, inplace=True)
        feature = scaler.transform(feature.values.reshape(-1, 1))

        # Set missing values nan again and update dataframe
        feature[missing_indices] = np.nan
        new_features[col] = feature

    return data.replace(X=new_features)
