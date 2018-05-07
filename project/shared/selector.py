"""
    Base class for transformers which can be used for dim_reduction in pipeline 
"""
import sys
from abc import ABC, abstractmethod
from project.utils.assertions import assert_data, assert_df, assert_series


class Selector(ABC):
    def __init__(self, data, **kwargs):
        """
        Base class for feature selection approaches

        Arguments:
            data {Data} -- Data object
        """
        self.data = assert_data(data)
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

    def fit(self, X=None, y=None):
        """
        Fit a selector and store ranking

        Keyword Arguments:
            X {df} -- New feature matrix (default: {None})
            y {pd.series} -- New label vector (default: {None})
        """
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        # Update data if X and y are passed (usually this only happens when using a Pipeline)
        if not X is None and not y is None:
            X = assert_df(X).reset_index(drop=True)
            y = assert_series(y).reset_index(drop=True)
            self.data = self.data._replace(features=X, labels=y)

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
        return sorted(self.feature_importances.items(),
                      key=lambda k_v: k_v[1], reverse=True)

    def get_selected_features(self):
        """
        Returns k best features
        """
        return [kv[0] for kv in self.ranking[:self.params["k"]]]

    def transform(self, X=None):
        """
        Returns dataframe with best features

        Keyword Arguments:
            X {df} -- New features matrix (default: {None})
        """
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")

        # Update features when X not none (mostly in Pipeline)
        if not X is None:
            X = assert_df(X).reset_index(drop=True)
            self.data = self.data._replace(features=X)
        return self.data.features[self.selected_features]

    def fit_transform(self, X=None, y=None):
        """
        Fit selector and return transformed data

        Keyword Arguments:
            X {df} -- New features matrix (default: {None})
            y {df.series} -- new labels vector (default: {None})
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        """
        Returns boolean vector of selected features
        """
        return self.data.features.columns.isin(self.selected_features)

    def get_params(self, deep=False):
        """
        Returns params used in sklearn to copy objects

        Keyword Arguments:
            deep {bool} -- Deep copy (default: {False})
        """
        return {
            "data": self.data,
            **self.params,
        }
