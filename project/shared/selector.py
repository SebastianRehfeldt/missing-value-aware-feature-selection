"""
    Base class for transformers which can be used for dim_reduction in pipeline
"""
import sys
from abc import ABC, abstractmethod
from project import Data
from project.utils import assert_data, assert_df, assert_series, assert_l_type


class Selector(ABC):
    def __init__(self, f_types, l_type, shape, **kwargs):
        """
        Base class for feature selection approaches

        Arguments:
            f_types {pd.Series} -- Series containing feature types
            l_type {str} -- Type of label
            shape {tuple} -- Tuple containing the shape of features
        """
        self.f_types = assert_series(f_types)
        self.l_type = assert_l_type(l_type)
        self.shape = shape
        self.is_fitted = False
        self._init_parameters(kwargs)

    @abstractmethod
    def _init_parameters(self, parameters):
        """
        Initialize parameters for selector

        Arguments:
            parameters {dict} -- Parameter dict
        """
        self.params = {}
        raise NotImplementedError("subclasses must override _init_parameters")

    @abstractmethod
    def calculate_feature_importances(self):
        """
        Calculate feature importances
        """
        raise NotImplementedError(
            "subclasses must override calculate_feature_importances")

    def fit(self, X, y):
        """
        Fit a selector and store ranking

        Keyword Arguments:
            X {df} -- Feature matrix
            y {pd.series} -- New label vector
        """
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        X = assert_df(X).reset_index(drop=True)
        y = assert_series(y).reset_index(drop=True)
        data = Data(X, y, self.f_types, self.l_type, X.shape)
        self.data = assert_data(data)

        # Calculate importanes and store ranking and selected features
        self.feature_importances = self.calculate_feature_importances()
        self.ranking = self.get_ranking()
        self.selected_features = self.get_selected_features()
        self.is_fitted = True
        return self

    def get_ranking(self):
        """
        Return features sorted by their importances
        """
        return sorted(
            self.feature_importances.items(),
            key=lambda k_v: k_v[1],
            reverse=True)

    def get_selected_features(self):
        """
        Returns k best features
        """
        return [kv[0] for kv in self.ranking[:self.params["k"]]]

    def transform(self, X):
        """
        Returns dataframe with best features

        Keyword Arguments:
            X {df} -- Feature matrix
        """
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")

        X = assert_df(X).reset_index(drop=True)
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit selector and return transformed data

        Keyword Arguments:
            X {df} -- Feature matrix
            y {df.series} -- New label vector
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        """
        Returns boolean vector of selected features
        """
        return self.data.X.columns.isin(self.selected_features)

    def get_params(self, deep=False):
        """
        Returns params used in sklearn to copy objects

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "f_types": self.f_types,
            "l_type": self.l_type,
            "shape": self.shape,
            **self.params,
        }
