import numpy as np
import pandas as pd
from .rar_params import RaRParams
from .rar_utils import RaRUtils
from .hics import HICS
from .optimizer import deduce_relevances


class RaR(RaRParams, RaRUtils):
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
        mr_boost = 1 + np.mean(self.missing_rates)
        n_iterations = int(self.params["contrast_iterations"] * mr_boost)
        self.params["contrast_iterations"] = n_iterations

    def _fit(self):
        self._set_nans()
        self._set_nan_corr()
        self.missing_rates = self.nans.sum() / self.data.shape[0]
        self._increase_iterations()
        self.scores_1d = pd.Series(np.zeros(len(self.names)), index=self.names)
        self.hics = HICS(self.data, self.nans, self.missing_rates,
                         **self.params)
        super()._fit()

    def _get_p_for_target(self, open_features, subspace):
        p = None
        if self.params["active_sampling"]:
            p = np.ones(len(open_features))
            if self.params.get("active_sampling_rel", False):
                p += self.scores_1d[open_features].values * 5

            if self.params.get("active_sampling_corr", False):
                corr = self.nan_corr.loc[subspace, open_features].min()
                p += (1 - corr) * 5
            p /= np.sum(p)
        return p

    def _get_targets(self, subspace):
        targets = []
        if self.params["redundancy_approach"] == "tom":
            open_features = [n for n in self.names if n not in subspace]
            n_targets = min(len(open_features), self.params["n_targets"])
            p = self._get_p_for_target(open_features, subspace)
            targets = np.random.choice(open_features, n_targets, False, p)
        return targets

    def _evaluate_subspace(self, subspace):
        targets = self._get_targets(subspace)
        rel, red_s, is_empty = self.hics.evaluate_subspace(subspace, targets)

        if is_empty:
            rel, red_s, targets = (0, [], [])

        if len(subspace) == 1 and self.params["active_sampling_rel"]:
            if rel > self.scores_1d[subspace[0]]:
                self.scores_1d[subspace[0]] = rel

        return {
            "relevance": rel,
            "redundancies": red_s,
            "targets": targets,
        }

    def _boost_values(self):
        alpha = self.params["boost_value"]
        if alpha > 0:
            for key, value in self.relevances.items():
                boost = self.hics.get_boost(key)[0]
                self.relevances[key] = (1 - alpha) * value + alpha * boost

    def _get_best_subsets(self):
        sets = [d for d in self.score_map if len(d['features']) > 1]
        n_subsets = int(0.1 * self.params["n_subspaces"])
        n = max(20, min(100, n_subsets))
        return sorted(sets, key=lambda k: k["score"]["relevance"])[-n:]

    def _create_result(self, key):
        return {
            "features": [key],
            "score": {
                "relevance": self.scores_1d[key],
                "redundancies": [0],
                "targets": [],
            }
        }

    def _collect_last_open(self, open_fs):
        results = []
        for key in open_fs:
            res = self._evaluate_subspace([key])
            self.scores_1d[key] = res["relevance"]
            results.append({"features": [key], "score": res})
        return results

    def _collect_iteractions(self, d):
        results, features = [], d["features"]
        if self.params["boost_inter"] > 0:
            cum_rel = np.sum(self.scores_1d[features])
            if 1.5 * cum_rel <= d["score"]["relevance"]:
                self.interactions.append(features)
                d["score"]["relevance"] *= np.sqrt(len(features))
                results.append(d)
        return results

    def _collect_active_samples(self):
        results = []
        if self.params["active_sampling"]:
            open_fs = list(self.scores_1d.index[(self.scores_1d == 0).values])
            for d in self._get_best_subsets():
                # evaluate open features and add to results and 1d scores
                intersection = set(open_fs).intersection(set(d["features"]))
                for key in intersection:
                    self.scores_1d[key] = self.hics.evaluate_subspace([key])[0]
                    results.append(self._create_result(key))
                    open_fs.remove(key)

                results.extend(self._collect_iteractions(d))
            results.extend(self._collect_last_open(open_fs[:10]))
        return results

    def _deduce_feature_importances(self, knowledgebase):
        # apply active sampling and deduce feature relevances
        kb = knowledgebase + self._collect_active_samples()
        reg = self.params["regularization"]
        self.relevances = deduce_relevances(self.names, kb, reg)

        self._boost_values()
        if self.params["n_targets"] == 0:
            return self.get_sorted_relevances()
        return self.get_final_ranking(knowledgebase)
