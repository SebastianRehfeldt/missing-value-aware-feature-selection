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


def _deduce_redundancy(samples, selected_features):
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


def calculate_ranking(relevances, redundancies, names):
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
        for f in open_features:
            red = _deduce_redundancy(redundancies[f], selected)
            score = _combine_scores(relevances[f], red)
            if score >= best_score:
                best_score, best_feature = deepcopy(score), deepcopy(f)

        ranking[best_feature] = best_score
        open_features.remove(best_feature)
    return ranking


def create_subspace(best_feature, selected):
    n_max = min(2, len(selected))
    n_choose = np.random.choice(range(n_max), 1)[0]
    subspace = list(np.random.choice(selected, n_choose, False))

    # make sure the last feature is inside subspace
    if best_feature not in subspace:
        subspace.append(best_feature)
    return subspace


def calculate_ranking2(hics, relevances, names, n_targets):
    best = sorted(relevances.items(), key=lambda k_v: k_v[1], reverse=True)[0]
    best_feature = best[0]

    ranking = {}
    ranking[best_feature] = _combine_scores(best[1], 0)

    open_features = deepcopy(names)
    open_features.remove(best_feature)

    # stepwise add features
    max_redundancies = {feature: 0 for feature in open_features}
    max_calculations = max(10, int(np.sqrt(len(open_features))))
    if max_calculations * len(open_features) * n_targets > 10000:
        max_calculations = 5
        n_targets = 1

    while len(open_features) > 0:
        selected = list(ranking.keys())

        if len(selected) <= max_calculations:
            n = min(n_targets, len(selected))
            redundancies = np.zeros((n, len(open_features)))

            # compute redundancies to previous features using n subspaces
            for i in range(n):
                subspace = create_subspace(best_feature, selected)
                slices, lengths = hics.combine_slices(subspace)
                redundancies[i, :] = hics.compute_partial_redundancies(
                    slices, lengths, open_features)

            redundancies = np.max(redundancies, axis=0)
            for i, feature in enumerate(open_features):
                redundancy = max(redundancies[i], max_redundancies[feature])
                max_redundancies[feature] = redundancy
                redundancies[i] = redundancy
        else:
            redundancies = [max_redundancies[f] for f in open_features]

        combined_scores = [
            _combine_scores(relevances[f], redundancies[i])
            for i, f in enumerate(open_features)
        ]

        best_index = np.argmax(combined_scores)
        best_feature = open_features[best_index]
        ranking[best_feature] = combined_scores[best_index]
        open_features.remove(best_feature)

    return ranking
