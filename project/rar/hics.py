import numpy as np
import pandas as pd
from .hics_utils import HICSUtils
from .contrast import calculate_contrasts
from .slicing import prune_slices


class HICS(HICSUtils):
    def get_relevance(self, slices, lengths):
        y = self.data.y
        l_type = self.data.l_type
        indices = self.label_indices
        cache = self._create_cache(y, l_type, slices, lengths, indices)
        relevances = calculate_contrasts(cache)
        rel = 1 - np.exp(-1 * np.mean(relevances))
        return (rel, np.std(relevances))

    def get_redundancies(self, slices, lengths, targets, T=None):
        redundancies = []
        for target in targets:
            t_type = self.data.f_types[target]

            if self.params["approach"] == "imputation":
                # impute nans in redundancy target
                T = T if T is not None else self.X_complete
                t = T[target]
                t_slices = slices
            else:
                # remove samples with nans in target for redundancy calculation
                t_nans = self.nans[target]
                if np.sum(t_nans.values) > 0:
                    t = self.data.X[target][~t_nans]

                    min_samples = self.params["min_samples"]
                    t_slices = slices[:, ~t_nans]
                    t_slices, lengths = prune_slices(t_slices, min_samples)
                else:
                    t = self.data.X[target]
                    t_slices = slices

            if len(t_slices) <= self.params["min_slices"]:
                redundancies.append(0)
            else:
                cache = self._create_cache(t, t_type, t_slices, lengths)
                red_s = calculate_contrasts(cache)
                redundancies.append(np.mean(red_s))
        return redundancies

    def evaluate_subspace(self, subspace, targets=[]):
        # GET SLICES
        T = None
        if self.params["approach"] == "imputation":
            X, T, types = self._apply_imputation(subspace, targets)
            slices = self.get_slices(X, types)
            slices, lengths = prune_slices(slices, self.params["min_samples"])
        else:
            slices, lengths = self.get_cached_slices(subspace)

        # RETURN IF TOO FEW SLICES
        if len(slices) <= self.params["min_slices"]:
            return 0, [], True

        # COMPUTE RELEVANCE AND REDUNDANCIES
        rels, deviations = self.get_relevance(slices, lengths)
        n_resamples = self.params["resamples"]
        if deviations > 0.1 and len(subspace) == 1 and n_resamples > 0:
            n_retries = int(np.floor(deviations * n_resamples / 0.5)) + 1
            n_resamples = min(n_resamples, n_retries)
            resamples = np.zeros(n_resamples + 1)
            resamples[0] = rels
            for i in range(1, n_resamples + 1):
                new_slices, new_lengths = self.get_cached_slices(
                    subspace, False)
                resamples[i] = self.get_relevance(new_slices, new_lengths)[0]
            rels = np.mean(resamples)

        reds = self.get_redundancies(slices, lengths, targets, T)
        return rels, reds, False

    def get_boost(self, feature):
        X = self.data.X[feature].to_frame()
        types = pd.Series(self.data.f_types[feature], [feature])
        slices = self.get_slices(X, types, n=10, b=True)
        lengths = np.sum(slices, axis=1)
        return self.get_relevance(slices, lengths)
