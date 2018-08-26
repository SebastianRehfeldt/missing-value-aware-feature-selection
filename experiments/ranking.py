import pandas as pd
from time import clock
from copy import deepcopy
from collections import defaultdict

from project.utils import DataLoader
from project.utils.data import DataGenerator
from project.utils import introduce_missing_values, scale_data
from project.utils.imputer import Imputer
from project.utils.deleter import Deleter


def calc_mean_ranking(rankings):
    # mean ranking over complete datasets serves as gold standard on uci
    relevances = defaultdict(list)
    for algorithm, results in rankings['0.0'].items():
        for run in range(len(results)):
            mean = pd.DataFrame(results[run]).mean(0)
            mean = mean.sort_values(ascending=False)
            relevances[algorithm].append(mean)
    return relevances


def get_rankings(CONFIG, DATASET_CONFIG, ALGORITHMS):
    shuffle_seed = 0
    durations, rankings, relevances = {}, {}, []
    for i in range(CONFIG["n_runs"]):
        ### CREATE DATASET WHICH IS USED FOR EVALUATION ###
        if CONFIG["is_real_data"]:
            name = DATASET_CONFIG["name"]
            data_loader = DataLoader(ignored_attributes=["molecule_name"])
            data_original = data_loader.load_data(name, "arff")
            data_original.shuffle_rows(seed=CONFIG["seeds"][i])
            relevances = None
        else:
            params = DATASET_CONFIG.copy()
            if CONFIG["update_config"]:
                params.update(CONFIG["updates"][i])
            generator = DataGenerator(**params)
            generator.set_seed(CONFIG["seeds"][i])
            data_original, relevance_vector = generator.create_dataset()
            relevances.append(relevance_vector)

        data_original = scale_data(data_original)

        ### GATHER RESULTS FOR SPECIFIC MISSING RATE ###
        for mr in CONFIG["missing_rates"]:
            durations_run = defaultdict(list)
            rankings_run = defaultdict(list)

            ### ADD MISSING VALUES TO DATASET (MULTIPLE TIMES) ###
            for j in range(CONFIG["n_insertions"]):
                data_orig = deepcopy(data_original)
                seed = CONFIG["seeds"][j]
                data_orig = introduce_missing_values(data_orig, mr, seed=seed)
                data_orig.shuffle_columns(shuffle_seed)
                shuffle_seed += 1

                for key, algorithm in ALGORITHMS.items():
                    ### GET RANKING USING SELECTOR ###
                    data = deepcopy(data_orig)

                    start = clock()
                    if algorithm.get("should_impute", False):
                        imputer = Imputer(data.f_types, algorithm["strategy"])
                        data = imputer.complete(data)

                    if algorithm.get("should_delete", False):
                        deleter = Deleter()
                        data = deleter.remove(data)

                    selector = algorithm["class"](
                        data.f_types,
                        data.l_type,
                        data.shape,
                        **algorithm["config"],
                    )
                    # fix for global deletion
                    if data.shape[0] >= 30:
                        selector.fit(data.X, data.y)
                    ranking = selector.get_ranking()
                    duration = clock() - start

                    rankings_run[key].append(dict(ranking))
                    durations_run[key].append(duration)

            # Update combined results
            if i == 0:
                durations[mr] = defaultdict(list)
                rankings[mr] = defaultdict(list)

            for key in ALGORITHMS.keys():
                durations[mr][key].append(durations_run[key])
                rankings[mr][key].append(rankings_run[key])

            print("Finished missing rate {:.1f}".format(mr), flush=True)
        print("Finished run {:d}".format(i + 1), flush=True)

    if not CONFIG["is_real_data"]:
        relevances = pd.DataFrame(relevances).T
    return rankings, durations, relevances
