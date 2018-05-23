"""
    RaR class for feature selection
"""
import numpy as np
from project.base import Subspacing


class RaR(Subspacing):
    def _init_parameters(self, **kwargs):
        """
        Initialize params

        Arguments:
            parameters {dict} -- Parameter dict
        """
        super()._init_parameters(**kwargs)
        max_size = min(5, self.shape[1] - 1)
        self.params["subspace_size"] = kwargs.get("subspace_size",
                                                  (1, max_size))

    def _complete(self, names, types, target):
        if self.params.get("approach", "deletion") == "deletion":
            indices = self.data.X[names + [target]].notnull().apply(
                all, axis=1)
            new_X = self.data.X[names][indices]
            new_t = self.data.X[target][indices]
            new_y = self.data.y[indices]

        return new_X, new_y, new_t

    def _evaluate_subspace(self, X, types):
        """
        Evaluate a subspace using hics measure

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """
        from .contrast import calculate_contrast
        from .slicing import get_slices
        import random

        names = X.columns.tolist()
        open_features = [
            name for name in self.data.X.columns.tolist() if name not in names
        ]
        target = random.choice(open_features)

        # TODO: deletion with target removes more samples than neccessary
        new_X, new_y, new_t = self._complete(names, types, target)

        relevances = []
        redundancies = []
        # TODO make param for #iterations
        # TODO values from paper
        n_select = int(0.8 * new_X.shape[0])
        for i in range(10):
            slice_vector = get_slices(new_X, types, n_select)
            relevances.append(
                calculate_contrast(new_y, self.l_type, slice_vector))
            redundancies.append(
                calculate_contrast(new_t, self.f_types[target], slice_vector))

        return {
            "relevance": np.mean(relevances),
            "redundancy": np.mean(redundancies),
            "target": target,
        }

    def _deduce_feature_importances(self, knowledgebase):
        """
        Deduce single feature importances based on subspace results

        Arguments:
            knowledgebase {list} -- List of subspace results
        """
        # TODO: deduce scores

        from pprint import pprint
        pprint(knowledgebase)
        return {}
