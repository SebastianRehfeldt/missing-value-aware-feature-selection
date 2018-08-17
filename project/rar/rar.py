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

    def _evaluate_subspace(self, subspace):
        targets = []
        if self.params["redundancy_approach"] == "tom":
            open_features = [n for n in self.names if n not in subspace]
            n_targets = min(len(open_features), self.params["n_targets"])

            p = None
            if self.params["active_sampling"]:
                corr = self.nan_correlation.loc[subspace, open_features].min()
                p = self.scores_1d[open_features].values * 5 + 1
                p += (1 - corr) * 5
                p /= np.sum(p)
            targets = np.random.choice(open_features, n_targets, False, p)

        results = self.hics.evaluate_subspace(subspace, targets)
        rel, red_s, is_empty, deviation = results

        if len(subspace) == 1 and self.params["active_sampling"]:
            if rel > self.scores_1d[subspace[0]]:
                self.scores_1d[subspace[0]] = rel

        if is_empty:
            rel, red_s, targets, deviation = 0, [], [], 0

        return {
            "relevance": rel,
            "deviation": deviation,
            "redundancies": red_s,
            "targets": targets,
        }

    def _deduce_feature_importances(self, knowledgebase):
        """
        Deduce single feature importances based on subspace results

        Arguments:
            knowledgebase {list} -- List of subspace results
        """
        if self.params["active_sampling"]:
            results = []
            open_fs = set(self.scores_1d.index[(self.scores_1d == 0).values])

            m = [d for d in self.score_map if len(d['features']) > 1]
            n = int(0.1 * self.params["n_subspaces"])
            n = max(20, min(100, n))
            m = sorted(
                m, key=lambda k: k["score"]["relevance"], reverse=True)[:n]

            for d in m:
                features = set(d["features"])
                rel = d["score"]["relevance"]
                intersection = list(open_fs.intersection(features))

                for key in intersection:
                    self.scores_1d[key] = self.hics.evaluate_subspace([key])[0]
                    open_fs.remove(key)

                if self.params["boost"] > 0:
                    cum_rel = np.sum(self.scores_1d[features])
                    if 1.5 * cum_rel <= rel:
                        self.interactions.append(features)
                        d["score"]["relevance"] *= np.sqrt(len(features))
                        results.append(d)

                    for key in intersection:
                        results.append({
                            "features": [key],
                            "score": {
                                "relevance": self.scores_1d[key],
                                "deviation": 0,
                                "redundancies": [0],
                                "targets": [],
                            }
                        })

            if 0 < len(open_fs) < 10:
                for key in open_fs:
                    res = self._evaluate_subspace([key])
                    self.scores_1d[key] = res["relevance"]
                    results.append({"features": [key], "score": res})

            kb = knowledgebase + results
            reg = self.params["regularization"]
            relevances = deduce_relevances(self.names, kb, reg)
        else:
            reg = self.params["regularization"]
            relevances = deduce_relevances(self.names, knowledgebase, reg)

        if self.params["boost"] > 0:
            for key, value in relevances.items():
                alpha = self.params["boost"]
                boost = self.hics.get_boost(key)
                relevances[key] = (1 - alpha) * value + alpha * boost[0]

        # return ranking based on relevances only
        n_targets = self.params["n_targets"]
        if n_targets == 0:
            return sorted(
                relevances.items(),
                key=lambda k_v: k_v[1],
                reverse=True,
            )

        # combine relevances with redundancies as done by tom or arvind
        if self.params["redundancy_approach"] == "tom":
            redundancies = sort_redundancies_by_target(knowledgebase)
            return get_ranking_tom(relevances, redundancies, self.names,
                                   self.nan_correlation,
                                   self.params["nullity_corr_boost"])

        return get_ranking_arvind(self.hics, relevances, self.names, n_targets,
                                  self.nan_correlation,
                                  self.params["nullity_corr_boost"])
