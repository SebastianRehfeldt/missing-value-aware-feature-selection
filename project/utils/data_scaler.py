import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_data(data, method="standard"):
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

    return data._replace(features=new_features)
