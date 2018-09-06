import numpy as np
import pandas as pd
from .hics_slicing import HICSSlicing


class HICSUtils(HICSSlicing):
    def get_cached_slices(self, subspace, use_cache=True):
        dim = len(subspace)
        if self.params["cache_enabled"]:
            if use_cache:
                slices = [self.slices[subspace[i]][dim] for i in range(dim)]
                slices = self.combine_slices(slices)
            else:
                # shuffle slices instead of creating new for multi-d spaces
                # shuffling does not change contrast when having 1 dim only
                if len(subspace) > 1:
                    slices = [None] * len(subspace)
                    for i in range(dim):
                        slices[i] = self.slices[subspace[i]][dim]
                        np.random.shuffle(slices[i])
                    slices = self.combine_slices(slices)
                else:
                    slices = self.get_slices(subspace)
        else:
            slices = self.get_slices(subspace)

        if self.params["approach"] == "partial":
            max_nans = int(np.floor(dim / 2))
            nan_sums = np.sum(self.nans[subspace].values, axis=1)
            slices[:, nan_sums > max_nans] = False
        return self.prune_slices(slices)

    def _apply_imputation(self, subspace, targets):
        from project.utils.imputer import Imputer

        types = self.data.f_types[subspace]
        target_types = self.data.f_types[targets]
        all_types = pd.concat([types, target_types], axis=0)
        cols = np.hstack((subspace, targets))

        imputer = Imputer(all_types, self.params["imputation_method"])
        X_complete = imputer._complete(self.data.X[cols])
        return X_complete[subspace], X_complete[targets]

    def _create_cache(self, y, slices):
        is_label = self.data.y.name == y.name
        if is_label:
            y_type = self.data.l_type
            sorted_indices = self.label_indices
            sorted_y = self.label_sorted
        else:
            y_type = self.data.f_types[y.name]
            sorted_indices = np.argsort(y.values)
            sorted_y = y.values[sorted_indices]

        cache = {
            "type": y_type,
            "sorted": sorted_y,
            "slices": slices[:, sorted_indices],
        }

        if y_type == "nominal":
            if is_label:
                counts = self.label_cache[1]
            else:
                counts = np.unique(sorted_y, return_counts=True)[1]
            cache.update({"counts": counts})
        return cache
