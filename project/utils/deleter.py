"""
    Row deleter
"""
import numpy as np
from project.utils import assert_data


class Deleter():
    def __init__(self):
        """
        Deleter class for removing samples with missing values
        """

    def remove(self, data):
        """
        Complete features
        """
        data = assert_data(data)

        indices = data.X.notnull().apply(all, axis=1)
        nominal_indices = (data.X == "?").apply(any, axis=1)

        indices = np.logical_and(indices, ~nominal_indices)
        new_X = data.X[indices]
        new_y = data.y[indices]
        return data.replace(copy=True, X=new_X, y=new_y, shape=new_X.shape)
