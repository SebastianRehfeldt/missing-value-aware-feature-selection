from copy import deepcopy
from collections import defaultdict


def sort_redundancies_by_target(knowledgebase):
    redundancies = defaultdict(list)
    for subset in knowledgebase:
        features = subset["features"]
        target = subset["score"]["target"]
        redundancy = subset["score"]["redundancy"]
        redundancies[target].append((features, redundancy))
    return redundancies


def _combine_scores(rel, red):
    # TODO redundancy might be too important
    return 2 * (1 - red) * rel / ((1 - red) + rel)


def _calculate_redundancy(samples, selected_features):
    # TODO: compute intersections for all samples only once
    # TODO: vectorize search for admissables and justified
    # TODO: any way for caching?

    # get admissables
    admissables = [
        s for s in samples
        if len(set(s[0]).intersection(selected_features)) > 0
    ]

    # get minimum redundancy of justified samples
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


def calculate_ranking(relevances, redundancies, names):
    # TODO: complete sorting not necessary
    best = sorted(relevances.items(), key=lambda k_v: k_v[1], reverse=True)[0]

    ranking = {}
    ranking[best[0]] = _combine_scores(best[1], 0)

    open_features = deepcopy(names)
    open_features.remove(best[0])

    # stepwise add features
    while len(open_features) > 0:
        # TODO vectorize
        best_score, best_feature = 0, None
        for f in open_features:
            # deduce redundancy of feature to previous feature
            red = _calculate_redundancy(redundancies[f], set(ranking.keys()))
            score = _combine_scores(relevances[f], red)

            if score > best_score:
                best_score, best_feature = score, f

        ranking[best_feature] = best_score
        # TODO causes a bug sometimes
        open_features.remove(best_feature)
    return ranking
