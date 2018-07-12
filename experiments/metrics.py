import numpy as np


def calculate_cg(gold_ranking, ranking):
    CG = np.zeros(len(ranking))
    i, current_cg = 0, 0
    for feature in ranking:
        current_cg += gold_ranking[feature]
        CG[i] = current_cg
        i += 1
    return CG


def calculate_dcg(gold_ranking, ranking):
    i, DCG = 2, 0
    for feature in ranking:
        DCG += gold_ranking[feature] / np.log2(i)
        i += 1
    return DCG


def calculate_ndcg(gold_ranking, ranking):
    DCG = calculate_dcg(gold_ranking, ranking)

    # calculate ideal discounted cumulative gain for normalization
    i, IDCG = 2, 0
    optimal_ranking = gold_ranking.sort_values(ascending=False)
    for score in optimal_ranking:
        IDCG += score / np.log2(i)
        i += 1

    return DCG / IDCG


def calculate_sse(gold_ranking, ranking):
    return (gold_ranking - ranking).pow(2).sum()


def calculate_mse(gold_ranking, ranking):
    sse = calculate_sse(gold_ranking, ranking)
    return sse / len(ranking)