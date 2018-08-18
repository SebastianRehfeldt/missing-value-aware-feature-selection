import numpy as np
import pandas as pd
from .hics_params import HICSParams
from .slicing import get_slices, combine_slices, prune_slices


class HICSUtils(HICSParams):
    def get_slices(self, X, types, cache=None, n=None, d=None, b=False):
        options = self.slice_options.copy()
        options["n_select"] = n or self.n_select_d[X.shape[1]]
        options["alpha"] = self.alphas_d[X.shape[1]]
        options["d"] = d or X.shape[1]
        options["boost"] = b
        if self.params["weight_approach"] == "new":
            options["X_complete"] = self.X_complete[X.columns]
        return get_slices(X, types, cache, **options)

    def get_cached_slices(self, subspace, use_cache=True):
        dim = len(subspace)

        if self.params["cache_enabled"] and use_cache:
            slices = [self.slices[subspace[i]][dim] for i in range(dim)]
            slices = combine_slices(slices).copy()
        else:
            X = self.data.X[subspace]
            types = self.data.f_types[subspace]
            slices = self.get_slices(X, types)

        min_samples = self.params["min_samples"]
        if self.params["approach"] in ["partial", "fuzzy"]:
            max_nans = int(np.floor(dim / 2))
            if self.params["approach"] == "fuzzy":
                max_nans = dim

            nan_sums = np.sum(self.nans[subspace].values, axis=1)
            slices[:, nan_sums > max_nans] = False
        return prune_slices(slices, min_samples)

    def _apply_imputation(self, subspace, targets):
        from project.utils.imputer import Imputer

        types = self.data.f_types[subspace]
        target_types = self.data.f_types[targets]
        all_types = pd.concat([types, target_types], axis=0)
        cols = np.hstack((subspace, targets))

        imputer = Imputer(all_types, self.params["imputation_method"])
        X_complete = imputer._complete(self.data.X[cols])
        return X_complete[subspace], X_complete[targets], types

    def _create_cache(self, y, y_type, slices, lengths, cached_indices=None):
        if cached_indices is None:
            sorted_indices = np.argsort(y.values)
        else:
            sorted_indices = cached_indices

        sorted_y = y.values[sorted_indices]
        cache = {
            "type": y_type,
            "lengths": lengths,
            "sorted": sorted_y,
            "slices": slices[:, sorted_indices],
        }

        if y_type == "nominal":
            if self.data.y.name == y.name:
                values, counts = self.label_cache
            else:
                values, counts = np.unique(sorted_y, return_counts=True)

            cache.update({
                "values": values,
                "counts": counts,
            })
        return cache
