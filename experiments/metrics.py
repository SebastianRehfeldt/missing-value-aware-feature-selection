import numpy as np
import pandas as pd
from collections import defaultdict


def calc_cg(gold_ranking, ranking, use_position=False):
    CG = np.zeros(len(gold_ranking))
    current_cg = 0
    sorted_values = gold_ranking.sort_values(ascending=False)
    sum_of_indices = 0
    for i in range(len(gold_ranking)):
        sum_of_indices += 1 / (i + 2)

    for i, feature in enumerate(ranking):
        score = gold_ranking[feature]
        if use_position:
            score = 1 / (sorted_values.index.get_loc(feature) + 2)
            score /= sum_of_indices
        current_cg += score
        CG[i] = current_cg
    CG[CG == 0] = current_cg
    return CG


def calc_dcg(gold_ranking, ranking, use_position=False):
    i, DCG = 2, 0
    sorted_values = gold_ranking.sort_values(ascending=False)
    for feature in ranking:
        if use_position:
            score = 1 / (sorted_values.index.get_loc(feature) + 2)
        else:
            score = gold_ranking[feature]
        DCG += score / np.log2(i)
        i += 1
    return DCG


def calc_ndcg(gold_ranking, ranking, use_position=False):
    DCG = calc_dcg(gold_ranking, ranking, use_position)

    # calculate ideal discounted cumulative gain for normalization
    i, IDCG = 2, 0
    optimal_ranking = gold_ranking.sort_values(ascending=False)
    for score in optimal_ranking:
        if use_position:
            score = 1 / i
        IDCG += score / np.log2(i)
        i += 1
    return DCG / IDCG


def calc_sse(gold_ranking, ranking):
    return (gold_ranking - ranking).pow(2).sum()


def calc_mse(gold_ranking, ranking):
    sse = calc_sse(gold_ranking, ranking)
    return sse / len(ranking)


def compute_statistics(rankings, relevances):
    d = defaultdict(dict)
    first_key = list(rankings.keys())[0]
    cg_means, cg_deviations = d.copy(), d.copy()
    ndcg_means, ndcg_deviations = d.copy(), d.copy()
    cg_means_pos, cg_deviations_pos = d.copy(), d.copy()
    ndcg_means_pos, ndcg_deviations_pos = d.copy(), d.copy()
    sse_means, sse_deviations = d.copy(), d.copy()

    for missing_rate in rankings.keys():
        for key in rankings[missing_rate].keys():
            ranking = rankings[missing_rate][key]

            # The mean and std are calculated over all datasets and insertions
            # Run means a new dataset and i indicates multiple insertions
            cgs, cgs_pos, ndcgs, ndcgs_pos, sses = [], [], [], [], []
            for run in range(len(ranking)):
                for i in range(len(ranking[run])):
                    # use ranking on complete data as gold ranking for uci
                    if isinstance(relevances, pd.DataFrame):
                        gold_scores = relevances[str(run)]
                        complete_scores = rankings[first_key][key][run][i]
                        complete_scores = pd.Series(complete_scores)
                    else:
                        complete_scores = relevances[key][run]
                        gold_scores = complete_scores / complete_scores.sum()

                    # CG and NDCG
                    t = 1e-4
                    scores = [k for k, v in ranking[run][i].items() if v > t]
                    CG = calc_cg(gold_scores, scores)
                    cgs.append(CG)
                    NDCG = calc_ndcg(gold_scores, scores)
                    ndcgs.append(NDCG)

                    CG_POS = calc_cg(gold_scores, scores, True)
                    cgs_pos.append(CG_POS)
                    NDCG_POS = calc_ndcg(gold_scores, scores, True)
                    ndcgs_pos.append(NDCG_POS)

                    # SSE
                    scores = pd.Series(ranking[run][i])
                    SSE = calc_sse(complete_scores, scores)
                    sses.append(SSE)

            cg_means[missing_rate][key] = np.mean(cgs, axis=0)
            cg_deviations[missing_rate][key] = np.std(cgs, axis=0)

            ndcg_means[missing_rate][key] = np.mean(ndcgs)
            ndcg_deviations[missing_rate][key] = np.std(ndcgs)

            cg_means_pos[missing_rate][key] = np.mean(cgs_pos, axis=0)
            cg_deviations_pos[missing_rate][key] = np.std(cgs_pos, axis=0)

            ndcg_means_pos[missing_rate][key] = np.mean(ndcgs_pos)
            ndcg_deviations_pos[missing_rate][key] = np.std(ndcgs_pos)

            sse_means[missing_rate][key] = np.mean(sses)
            sse_deviations[missing_rate][key] = np.std(sses)

    ndcg_means = pd.DataFrame(ndcg_means).T
    ndcg_deviations = pd.DataFrame(ndcg_deviations).T
    ndcg_means_pos = pd.DataFrame(ndcg_means_pos).T
    ndcg_deviations_pos = pd.DataFrame(ndcg_deviations_pos).T
    sse_means = pd.DataFrame(sse_means).T
    sse_deviations = pd.DataFrame(sse_deviations).T
    mse_means = sse_means / len(rankings[missing_rate][key])

    cgs = (cg_means, cg_deviations)
    ndcgs = (ndcg_means, ndcg_deviations)
    cgs_pos = (cg_means_pos, cg_deviations_pos)
    ndcgs_pos = (ndcg_means_pos, ndcg_deviations_pos)
    sses = (sse_means, sse_deviations)
    mses = (mse_means, sse_deviations)
    return cgs, ndcgs, cgs_pos, ndcgs_pos, sses, mses
