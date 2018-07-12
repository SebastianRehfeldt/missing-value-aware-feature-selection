import numpy as np
import pandas as pd


def calc_cg(gold_ranking, ranking):
    CG = np.zeros(len(ranking))
    i, current_cg = 0, 0
    for feature in ranking:
        current_cg += gold_ranking[feature]
        CG[i] = current_cg
        i += 1
    return CG


def calc_dcg(gold_ranking, ranking):
    i, DCG = 2, 0
    for feature in ranking:
        DCG += gold_ranking[feature] / np.log2(i)
        i += 1
    return DCG


def calc_ndcg(gold_ranking, ranking):
    DCG = calc_dcg(gold_ranking, ranking)

    # calculate ideal discounted cumulative gain for normalization
    i, IDCG = 2, 0
    optimal_ranking = gold_ranking.sort_values(ascending=False)
    for score in optimal_ranking:
        IDCG += score / np.log2(i)
        i += 1

    return DCG / IDCG


def calc_sse(gold_ranking, ranking):
    return (gold_ranking - ranking).pow(2).sum()


def calc_mse(gold_ranking, ranking):
    sse = calc_sse(gold_ranking, ranking)
    return sse / len(ranking)


def compute_statistics(rankings, relevances):
    first_key = list(rankings.keys())[0]
    cg_means, cg_deviations = {}, {}
    ndcg_means, ndcg_deviations = {}, {}
    sse_means, sse_deviations = {}, {}

    for missing_rate in rankings.keys():
        cg_means[missing_rate] = {}
        cg_deviations[missing_rate] = {}
        ndcg_means[missing_rate] = {}
        ndcg_deviations[missing_rate] = {}
        sse_means[missing_rate] = {}
        sse_deviations[missing_rate] = {}

        for key in rankings[missing_rate].keys():
            ranking = rankings[missing_rate][key]

            # The mean and std are calculated over all datasets and insertions
            # Run means a new dataset and i indicates multiple insertions
            cgs, ndcgs, sses = [], [], []
            for run in range(len(ranking)):
                gold_scores = relevances[str(run)]

                for i in range(len(ranking[run])):
                    # CG and NDCG
                    t = 1e-4
                    scores = [k for k, v in ranking[run][i].items() if v > t]
                    CG = calc_cg(gold_scores, scores)
                    cgs.append(CG)
                    NDCG = calc_ndcg(gold_scores, scores)
                    ndcgs.append(NDCG)

                    # SSE
                    complete_scores = rankings[first_key][key][run][i]
                    complete_scores = pd.Series(complete_scores)
                    scores = pd.Series(ranking[run][i])
                    SSE = calc_sse(complete_scores, scores)
                    sses.append(SSE)

            cg_means[missing_rate][key] = np.mean(cgs, axis=0)
            cg_deviations[missing_rate][key] = np.std(cgs, axis=0)

            ndcg_means[missing_rate][key] = np.mean(ndcgs)
            ndcg_deviations[missing_rate][key] = np.std(ndcgs)

            sse_means[missing_rate][key] = np.mean(sses)
            sse_deviations[missing_rate][key] = np.std(sses)

    ndcg_means = pd.DataFrame(ndcg_means).T
    ndcg_deviations = pd.DataFrame(ndcg_deviations).T
    sse_means = pd.DataFrame(sse_means).T
    sse_deviations = pd.DataFrame(sse_deviations).T
    mse_means = sse_means / len(rankings[missing_rate][key])
    return (cg_means, cg_deviations), (ndcg_means, ndcg_deviations), (
        sse_means, sse_deviations), (mse_means, sse_deviations)
