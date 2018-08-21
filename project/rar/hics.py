import numpy as np
from .hics_utils import HICSUtils
from .contrast import calculate_contrasts


class HICS(HICSUtils):
    def get_relevance(self, slices):
        cache = self._create_cache(self.data.y, slices)
        relevances = calculate_contrasts(cache)
        rel = 1 - np.exp(-1 * np.mean(relevances))
        return (rel, np.std(relevances))

    def compute_relevance(self, slices, subspace):
        rels, deviations = self.get_relevance(slices)

        n_resamples = self.params["resamples"]
        n_retries = int(np.floor(deviations * n_resamples / 0.5)) + 1
        n_resamples = min(n_resamples, n_retries)

        if len(subspace) == 1 and n_resamples > 1:
            resamples = np.zeros(n_resamples + 1)
            resamples[0] = rels
            for i in range(1, n_resamples + 1):
                new_slices = self.get_cached_slices(subspace, False)
                resamples[i] = self.get_relevance(new_slices)[0]
            rels = np.mean(resamples)
        return rels

    def get_redundancies(self, slices, targets, T=None):
        redundancies = []
        for target in targets:
            if self.params["approach"] == "imputation":
                # in arvinds approach take imputation on all fs
                T = self.X_complete if T is None else T
                t, t_slices = T[target], slices
            else:
                # remove samples with nans in target
                t_nans = self.nans[target]
                if self.nan_sums[target] > 0:
                    t = self.data.X[target][~t_nans]
                    t_slices = slices[:, ~t_nans]
                    t_slices = self.prune_slices(t_slices)
                else:
                    t, t_slices = self.data.X[target], slices

            if len(t_slices) <= self.params["min_slices"]:
                redundancies.append(0)
            else:
                cache = self._create_cache(t, t_slices)
                red_s = calculate_contrasts(cache)
                redundancies.append(np.mean(red_s))
        return redundancies

    def evaluate_subspace(self, subspace, targets=[]):
        # GET SLICES
        T = None
        if self.params["approach"] == "imputation":
            X, T = self._apply_imputation(subspace, targets)
            slices = self.get_slices(subspace, X=X)
            slices = self.prune_slices(slices)
        else:
            slices = self.get_cached_slices(subspace)

        # RETURN IF TOO FEW SLICES
        if len(slices) <= self.params["min_slices"]:
            return 0, [], True

        # COMPUTE RELEVANCE AND REDUNDANCIES
        rels = self.compute_relevance(slices, subspace)
        reds = self.get_redundancies(slices, targets, T)
        return rels, reds, False

    def get_boost(self, feature):
        slices = self.get_slices([feature], n=10, boost=True)
        return self.get_relevance(slices)
