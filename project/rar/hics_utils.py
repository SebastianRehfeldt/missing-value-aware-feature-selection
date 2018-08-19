import numpy as np
import pandas as pd
from .hics_slicing import HICSSlicing


class HICSUtils(HICSSlicing):
    def get_slices(self, subspace, cache=None, n=None, d=None, b=False,
                   X=None):
        options = {}
        options["n_select"] = n or self.n_select_d[len(subspace)]
        options["alpha"] = self.alphas_d[len(subspace)]
        options["d"] = d or len(subspace)
        options["boost"] = b
        if self.params["weight_approach"] == "new":
            options["X_complete"] = self.X_complete[self.data.X.columns]
        return self.compute_slices(subspace, cache, X, **options)

    def get_cached_slices(self, subspace, use_cache=True):
        dim = len(subspace)

        if self.params["cache_enabled"] and use_cache:
            slices = [self.slices[subspace[i]][dim] for i in range(dim)]
            slices = self.combine_slices(slices).copy()
        else:
            slices = self.get_slices(subspace)

        min_samples = self.params["min_samples"]
        if self.params["approach"] == "partial":
            max_nans = int(np.floor(dim / 2))
            nan_sums = np.sum(self.nans[subspace].values, axis=1)
            slices[:, nan_sums > max_nans] = False
        return self.prune_slices(slices, min_samples)

    def _apply_imputation(self, subspace, targets):
        from project.utils.imputer import Imputer

        types = self.data.f_types[subspace]
        target_types = self.data.f_types[targets]
        all_types = pd.concat([types, target_types], axis=0)
        cols = np.hstack((subspace, targets))

        imputer = Imputer(all_types, self.params["imputation_method"])
        X_complete = imputer._complete(self.data.X[cols])
        return X_complete[subspace], X_complete[targets]

    def _create_cache(self, y, y_type, slices, cached_indices=None):
        if cached_indices is None:
            sorted_indices = np.argsort(y.values)
        else:
            sorted_indices = cached_indices

        sorted_y = y.values[sorted_indices]
        cache = {
            "type": y_type,
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
