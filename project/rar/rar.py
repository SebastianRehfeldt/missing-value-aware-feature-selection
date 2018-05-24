"""
    RaR class for feature selection
"""
import random
import numpy as np
from collections import defaultdict

from project.base import Subspacing
from .optimizer import deduce_relevances
from .contrast import calculate_contrast
from .slicing import get_slices


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

    def _evaluate_subspace(self, X, types):
        """
        Evaluate a subspace using hics measure

        Arguments:
            X {df} -- Dataframe containing the features
            types {pd.series} -- Series containing the feature types
        """

        names = X.columns.tolist()
        open_features = [
            name for name in self.data.X.columns.tolist() if name not in names
        ]
        target = random.choice(open_features)

        # TODO: deletion with target removes more samples than neccessary
        # TODO: increase iterations when removing or imputing data
        # TODO: implement imputation
        new_X, new_y, new_t = self._complete(names, types, target)

        relevances = []
        redundancies = []
        # TODO make param for #iterations
        # TODO values from paper
        # TODO reduce slices by similarity
        n_select = int(0.8 * new_X.shape[0])
        for i in range(100):
            slice_vector = get_slices(new_X, types, n_select)
            relevances.append(
                calculate_contrast(new_y, self.l_type, slice_vector))
            redundancies.append(
                calculate_contrast(new_t, self.f_types[target], slice_vector))

        # TODO: normalization?

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
        relevances = deduce_relevances(self.data.X.columns.tolist(),
                                       knowledgebase)
        return self._calculate_ranking(knowledgebase, relevances)

    def _complete(self, names, types, target):
        if self.params.get("approach", "deletion") == "deletion":
            indices = self.data.X[names + [target]].notnull().apply(
                all, axis=1)
            new_X = self.data.X[names][indices]
            new_t = self.data.X[target][indices]
            new_y = self.data.y[indices]

        return new_X, new_y, new_t

    def _combine_scores(self, rel, red):
        return 2 * (1 - red) * rel / ((1 - red) + rel)

    def _calculate_redundancy(self, samples, selected_features):
        admissables = [
            s for s in samples
            if len(set(s[0]).intersection(selected_features)) > 0
        ]

        max_red = 0
        for sample in admissables:
            intersection = set(sample[0]).intersection(selected_features)

            min_red = 1
            for s in admissables:
                if intersection.issubset(set(s[0])) and s[1] < min_red:
                    min_red = s[1]

            # a sample is justified if there exists no other admissable which
            # contains the full intersection and has a lower redundancy score
            if sample[1] <= min_red and sample[1] > max_red:
                max_red = sample[1]

        return max_red

    def _calculate_ranking(self, knowledgebase, relevances):
        redundancies = defaultdict(list)
        for subset in knowledgebase:
            features = subset["features"]
            target = subset["score"]["target"]
            redundancy = subset["score"]["redundancy"]
            redundancies[target].append((features, redundancy))

        ranking = {}
        # add best feature to ranking with 0 redundancy
        best = sorted(
            relevances.items(), key=lambda k_v: k_v[1], reverse=True)[0]
        ranking[best[0]] = self._combine_scores(best[1], 0)

        open_features = self.data.X.columns.tolist()
        open_features.remove(best[0])
        # stepwise add features
        while len(open_features) > 0:
            best_score, best_feature = 0, None
            for feature in open_features:
                redundancy = self._calculate_redundancy(
                    redundancies[feature], set(ranking.keys()))
                score = self._combine_scores(relevances[feature], redundancy)
                if score > best_score:
                    best_score, best_feature = score, feature

            ranking[best_feature] = best_score
            open_features.remove(best_feature)

        return ranking
