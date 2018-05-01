import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_data(data, feature_types, method="standard"):
    scaler_dict = {
        "standard": StandardScaler,
    }
    scaler = scaler_dict.get(method, StandardScaler)()

    for col in range(data.shape[1]):
        if feature_types.iloc[col] == "nominal":
            continue

        # Fit scaler on complete data
        feature = data.iloc[:, col]
        complete_cases = feature.dropna()
        scaler.fit(complete_cases.reshape(-1, 1))

        # Get indices of missing values, fill with zeros, scale
        missing_indices = feature.isnull()
        feature.fillna(0, inplace=True)
        feature = scaler.transform(feature.reshape(-1, 1))

        # Set missing values nan again and update dataframe
        feature[missing_indices] = np.nan
        data.iloc[:, col] = feature
    return data
