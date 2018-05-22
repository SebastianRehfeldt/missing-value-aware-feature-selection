"""
    Row deleter
"""
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
        new_X = data.X.dropna()
        new_y = data.y[indices]
        return data.replace(copy=True, X=new_X, y=new_y, shape=new_X.shape)
