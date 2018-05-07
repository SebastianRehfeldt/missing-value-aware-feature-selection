"""
    Util functions for asserting the right usage of datatypes
"""
import pandas as pd
from project import Data


def assert_df(data):
    if isinstance(data, pd.DataFrame):
        return data

    try:
        df = data.to_frame()
        return df
    except:
        print("Data object is not a dataframe: ")
        print(data)
        raise


def assert_series(data):
    if isinstance(data, pd.Series):
        return data

    try:
        series = pd.Series(data)
        return series
    except:
        print("Data object is not a series: ")
        print(data)
        raise


def assert_types(types, col):
    if isinstance(types, pd.Series):
        return types

    try:
        series = pd.Series(types, [col])
        return series
    except:
        print("Selected feature types are not a series")
        print(types, col)
        raise


def assert_l_type(l_type):
    assert (isinstance(l_type, str)), "Label type is not a string"
    return l_type


def assert_data(data):
    assert (isinstance(data.features, pd.DataFrame)), "Features are not a df"
    assert (isinstance(data.labels, pd.Series)), "Labels are not a series"
    assert (isinstance(data.f_types, pd.Series)), "F_types are not a series"
    assert (isinstance(data.l_type, str)), "L_type is not string"

    assert (data.features.shape == data.shape), "Data got inconsistent shape"
    assert (data.shape[0] == len(data.labels)), "Inconsistent n_samples"
    assert (len(data.f_types) == data.shape[1]), "Inconsistent feature types"
    return data
