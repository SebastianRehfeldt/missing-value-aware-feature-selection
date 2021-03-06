"""
    Base class for transformers which can be used for dim_reduction in pipeline
"""
import sys
import numpy as np
from abc import ABC, abstractmethod
from project.utils import Data
from project.utils import assert_df, assert_series, assert_l_type


class Selector(ABC):
    @abstractmethod
    def _fit(self):
        raise NotImplementedError("subclasses must override _fit")

    def __init__(self, f_types, l_type, shape, **kwargs):
        self.f_types = assert_series(f_types)
        self.l_type = assert_l_type(l_type)
        self.shape = shape
        self._init_parameters(**kwargs)
        self.is_fitted = False
        self.names = self.f_types.index.tolist()
        self.feature_importances = {name: -1 for name in self.names}

    def _init_parameters(self, **kwargs):
        self.params = {
            "knn_neighbors": kwargs.get("knn_neighbors", 3),
            "mi_neighbors": kwargs.get("mi_neighbors", 6),
            "k": kwargs.get("k", min(10, int(self.shape[1] / 2))),
            "nominal_distance": kwargs.get("nominal_distance", 1),
            "use_cv": kwargs.get("use_cv", False),
            "eval_method": kwargs.get("eval_method", "mi"),
            "seed": kwargs.get("random_state"),
        }
        np.random.seed(self.params["seed"])

    def fit(self, X, y):
        if self.is_fitted:
            print("Selector is already fitted")
            return self

        X = assert_df(X).reset_index(drop=True)
        y = assert_series(y).reset_index(drop=True)
        data = Data(X, y, self.f_types, self.l_type, X.shape)
        self.data = data.shuffle_rows()

        self.domain = None
        if self.params["eval_method"] == "tree":
            self.domain = self.data.to_table().domain

        if self.params["eval_method"] == "mi":
            self.data = self.data.add_salt()

        self._fit()
        self.is_fitted = True
        return self

    def transform(self, X, k=None):
        """
        Returns dataframe with best features

        Keyword Arguments:
            X {df} -- Feature matrix
        """
        if not self.is_fitted:
            sys.exit("Classifier not fitted yet")

        X = assert_df(X).reset_index(drop=True)
        return X[self.get_selected_features(k)]

    def fit_transform(self, X, y, k=None):
        """
        Fit selector and return transformed data

        Keyword Arguments:
            X {df} -- Feature matrix
            y {df.series} -- New label vector
        """
        self.fit(X, y)
        return self.transform(X, k)

    def get_ranking(self):
        return sorted(
            self.feature_importances.items(),
            key=lambda k_v: k_v[1],
            reverse=True,
        )

    def get_selected_features(self, k=None):
        k = k or self.params["k"]
        return [kv[0] for kv in self.get_ranking()[:k]]

    def get_support(self, k=None):
        """
        Returns boolean vector of selected features
        """
        return self.data.X.columns.isin(self.get_selected_features(k))

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

    def set_params(self, **params):
        # remove Nones to enable None in grid search
        params = {k: v for k, v in params.items() if v is not None}
        self.params.update(params)
        self.is_fitted = False
