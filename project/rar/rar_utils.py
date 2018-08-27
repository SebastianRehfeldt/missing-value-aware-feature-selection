import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from .rar_params import RaRParams
from .hics import HICS


class RaRUtils(RaRParams):
    def _set_nans(self):
        self.nans = self.data.X.isnull()
        if self.params["create_category"]:
            mask = (self.data.X == "?")
            self.data.X = self.data.X.where(~mask, other="MISSING")
        else:
            nominal_nans = (self.data.X == "?")
            self.nans = np.logical_or(self.nans, nominal_nans)

    def _set_nan_corr(self):
        self.nan_corr = self.nans.corr()
        self.nan_corr.fillna(0, inplace=True)
        self.nan_corr = 1 - (1 + self.nan_corr) / 2

    def _increase_iterations(self):
        mr_boost = 1 + np.mean(self.missing_rates) * 2
        n_iterations = int(self.params["contrast_iterations"] * mr_boost)
        self.params["contrast_iterations"] = n_iterations

    def _fit(self):
        self._set_nans()
        self._set_nan_corr()
        self.nan_sums = self.nans.sum()
        self.missing_rates = self.nan_sums / self.data.shape[0]
        self._increase_iterations()
        self.scores_1d = pd.Series(np.zeros(len(self.names)), index=self.names)
        self.hics = HICS(self.data, self.nans, self.missing_rates,
                         self.nan_sums, **self.params)
        super()._fit()

    def get_sorted_relevances(self):
        return sorted(
            self.relevances.items(),
            key=lambda k_v: k_v[1],
            reverse=True,
        )

    def sort_redundancies_by_target(self, knowledgebase):
        redundancies = defaultdict(list)
        for subset in knowledgebase:
            features = subset["features"]
            targets = subset["score"]["targets"]
            scores = subset["score"]["redundancies"]
            for target, redundancy in zip(targets, scores):
                redundancies[target].append((features, redundancy))
        return redundancies

    def _combine_scores(self, rel, red):
        return 2 * (1 - red) * rel / ((1 - red) + rel)

    def _get_admissables(self, samples, selected):
        return [(s, set(s[0]).intersection(selected)) for s in samples
                if len(set(s[0]).intersection(selected)) > 0]

    def _get_maximum_redundancy(self, admissables):
        max_red = 0
        for sample, intersection in admissables:
            if sample[1] <= max_red:
                continue

            # compute minimum redundancy of samples which contain intersection
            # of the current sample and the selected features
            min_red = min([
                s[1] for s, _ in admissables
                if intersection.issubset(set(s[0]))
            ])

            # a sample is justified if there exists no other admissable which
            # contains the full intersection and has a lower redundancy score
            if sample[1] <= min_red:
                max_red = sample[1]
        return max_red

    def _deduce_redundancy(self, samples, selected_features):
        admissables = self._get_admissables(samples, selected_features)
        if len(samples) == 0:
            return 0
        if len(admissables) == 0:
            return np.mean([s[1] for s in samples])
        return self._get_maximum_redundancy(admissables)

    def _get_best_f_tom(self, selected_f, open_f, a):
        best_score = 0
        for f in open_f:
            red = self._deduce_redundancy(self.redundancies[f], selected_f)
            score = self._combine_scores(self.relevances[f], red)
            corr = self.nan_corr.loc[selected_f, f].max()
            score = (1 - a) * score + a * corr
            if score >= best_score:
                best_score, best_feature = score, f
        return best_feature, best_score

    def create_subspace(self, best_feature, selected):
        n_max = min(self.params["subspace_size"][1] - 1, len(selected))
        n_choose = np.random.choice(range(n_max), 1)[0]
        subspace = list(np.random.choice(selected, n_choose, False))
        if best_feature not in subspace:
            subspace.append(best_feature)
        return subspace

    def _get_max_calculations(self, open_features):
        # TODO: find good heuristic for max_calculations
        n_targets = self.params["n_targets"]
        max_calculations = max(30, int(np.sqrt(len(open_features))))
        if max_calculations * len(open_features) * n_targets > 1000:
            max_calculations, n_targets = 10, 1
        return max_calculations

    def _get_new_reds(self, selected_f, open_f, last_best):
        n = min(self.params["n_targets"], len(selected_f))
        reds = np.zeros((n, len(open_f)))
        for i in range(n):
            subspace = self.create_subspace(last_best, list(selected_f))
            slices = self.hics.get_cached_slices(subspace)
            reds[i, :] = self.hics.get_redundancies(slices, open_f)
        return np.max(reds, axis=0)

    def _update_reds(self, redundancies, open_f):
        for i, f in enumerate(open_f):
            redundancy = max(redundancies[i], self.max_redundancies[f])
            self.max_redundancies[f] = redundancy
            redundancies[i] = redundancy
        return redundancies

    def _get_reds(self, selected_f, open_f, last_best):
        if len(selected_f) <= self.max_calculations:
            redundancies = self._get_new_reds(selected_f, open_f, last_best)
            return self._update_reds(redundancies, open_f)
        else:
            return [self.max_redundancies[f] for f in open_f]

    def _get_best_f_arvind(self, selected_f, open_f, last_best, a):
        redundancies = self._get_reds(selected_f, open_f, last_best)
        combined_scores = np.asarray([
            self._combine_scores(self.relevances[f], redundancies[i])
            for i, f in enumerate(open_f)
        ])
        corr = self.nan_corr.loc[selected_f, open_f].max()
        combined_scores = (1 - a) * combined_scores + a * corr.values

        best_index = np.argmax(combined_scores)
        return open_f[best_index], combined_scores[best_index]

    def get_final_ranking(self, kb):
        ranking = {}
        a = self.params["boost_corr"]
        best = self.get_sorted_relevances()[0]
        ranking[best[0]] = (1 - a) * self._combine_scores(best[1], 0) + a

        open_f = deepcopy(self.names)
        open_f.remove(best[0])

        if self.params["redundancy_approach"] == "tom":
            self.redundancies = self.sort_redundancies_by_target(kb)
        else:
            self.max_redundancies = {feature: 0 for feature in open_f}
            self.max_calculations = self._get_max_calculations(open_f)

        # stepwise add features
        while len(open_f) > 0:
            selected_f = set(ranking.keys())
            if self.params["redundancy_approach"] == "tom":
                best = self._get_best_f_tom(selected_f, open_f, a)
            else:
                best = self._get_best_f_arvind(selected_f, open_f, best[0], a)

            ranking[best[0]] = best[1]
            open_f.remove(best[0])
        return ranking
