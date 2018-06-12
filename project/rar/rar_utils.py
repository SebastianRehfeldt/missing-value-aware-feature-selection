import numpy as np
from copy import deepcopy
from collections import defaultdict


def sort_redundancies_by_target(knowledgebase):
    redundancies = defaultdict(list)
    for subset in knowledgebase:
        features = subset["features"]
        targets = subset["score"]["targets"]
        scores = subset["score"]["redundancies"]
        for target, redundancy in zip(targets, scores):
            redundancies[target].append((features, redundancy))
    return redundancies


def _combine_scores(rel, red):
    return 2 * (1 - red) * rel / ((1 - red) + rel)


def _calculate_redundancy(samples, selected_features):
    if len(samples) == 0:
        return 0

    intersections = [
        set(s[0]).intersection(selected_features) for s in samples
    ]

    # get admissables
    admissables = [(s, intersections[i]) for i, s in enumerate(samples)
                   if len(intersections[i]) > 0]

    if len(admissables) == 0:
        return np.mean([s[1] for s in samples])

    # get minimum redundancy of justified samples
    max_red = 0
    for sample, intersection in admissables:
        if sample[1] <= max_red:
            continue

        # compute minimum redundancy of samples which contain the intersection
        # of the current sample and the selected features
        min_red = min(
            [s[1] for s, _ in admissables if intersection.issubset(set(s[0]))])

        # a sample is justified if there exists no other admissable which
        # contains the full intersection and has a lower redundancy score
        # samples with a lower score are skipped in the beginning of the loop
        if sample[1] <= min_red:
            max_red = sample[1]
    return max_red


def calculate_ranking(relevances, redundancies, redundancies_1d, names):
    best = sorted(relevances.items(), key=lambda k_v: k_v[1], reverse=True)[0]

    ranking = {}
    ranking[best[0]] = _combine_scores(best[1], 0)

    open_features = deepcopy(names)
    open_features.remove(best[0])

    # stepwise add features
    while len(open_features) > 0:
        best_score, best_feature = 0, None
        selected = set(ranking.keys())

        # deduce redundancies of features to previous feature
        if redundancies_1d is not None:
            reds_1d = redundancies_1d[list(selected)].T

        for f in open_features:
            red = _calculate_redundancy(redundancies[f], selected)
            if redundancies_1d is not None:
                max_red_1d = np.max(np.abs(reds_1d[f].values))
                red = np.mean([red, max_red_1d])

            score = _combine_scores(relevances[f], red)
            if score >= best_score:
                best_score, best_feature = deepcopy(score), deepcopy(f)

        ranking[best_feature] = best_score
        open_features.remove(best_feature)
    return ranking
