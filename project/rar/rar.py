import numpy as np
from .rar_params import RaRParams
from .optimizer import deduce_relevances
from .rar_utils import sort_redundancies_by_target
from .rar_utils import get_ranking_arvind, get_ranking_tom


class RaR(RaRParams):
    def _init_parameters(self, **kwargs):
        super()._init_parameters(**kwargs)
        self.hics = None
        self.interactions = []

    def _get_p_target(self, open_features, subspace):
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
            p = self._get_p_target(open_features, subspace)
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
            open_fs = self.scores_1d.index[(self.scores_1d == 0).values]

            for d in self._get_best_subsets():
                intersection = set(open_fs).intersection(set(d["features"]))

                # evaluate open features and add to results and 1d scores
                for key in intersection:
                    self.scores_1d[key] = self.hics.evaluate_subspace([key])[0]
                    results.append(self._create_result(key))
                    open_fs.remove(key)

                # boost interactions
                results.extend(self._collect_iteractions(d))

            # evaluate 10 or less open features (might be more)
            results.extend(self._collect_last_open(open_fs[:10]))
        return results

    def _deduce_feature_importances(self, knowledgebase):
        # apply active sampling and deduce feature relevances
        kb = knowledgebase + self._collect_active_samples()
        reg = self.params["regularization"]
        self.relevances = deduce_relevances(self.names, kb, reg)

        # boost values when they have high contrast in highest/lowest values
        self._boost_values()

        # return ranking based on relevances only
        if self.params["n_targets"] == 0:
            return sorted(
                self.relevances.items(),
                key=lambda k_v: k_v[1],
                reverse=True,
            )

        # combine relevances with redundancies as done by tom or arvind
        if self.params["redundancy_approach"] == "tom":
            redundancies = sort_redundancies_by_target(knowledgebase)
            return get_ranking_tom(self.relevances, redundancies, self.names,
                                   self.nan_corr,
                                   self.params["nullity_corr_boost"])

        return get_ranking_arvind(self.hics, self.relevances, self.names,
                                  self.params["n_targets"], self.nan_corr,
                                  self.params["nullity_corr_boost"])
