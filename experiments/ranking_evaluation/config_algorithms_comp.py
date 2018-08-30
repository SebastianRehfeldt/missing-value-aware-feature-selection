from project.rar.rar import RaR
from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.baseline import Baseline

SHARED = {
    "alpha": 0.02,
    "n_targets": 1,
    "boost_value": 0,
    "boost_inter": 0,
    "boost_corr": 0,
    "n_subspaces": 500,
    "subspace_size": (2, 2),
    "active_sampling": False,
}

ALGORITHMS = {
    "Baseline": {
        "class": Baseline,
        "config": {}
    },
    "RaR Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "RaR Fuzzy Distance": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "imputation_method": "soft",
            "dist_method": "distance",
            **SHARED
        }
    },
    "XGBoost": {
        "class": Embedded,
        "config": {}
    },
    "Relief": {
        "class": Orange,
        "config": {
            "eval_method": "relief"
        }
    },
    "FCBF": {
        "class": Orange,
        "config": {
            "eval_method": "fcbf"
        }
    },
    "Random Forest": {
        "class": Orange,
        "config": {
            "eval_method": "rf"
        }
    },
    "SFS + Tree": {
        "class": SFS,
        "config": {
            "eval_method": "tree"
        }
    },
    "RKNN": {
        "class": RKNN,
        "config": {}
    },
    "MI": {
        "class": Filter,
        "config": {
            "eval_method": "mi"
        }
    },
    "CFS": {
        "class": Ranking,
        "config": {
            "eval_method": "cfs"
        }
    },
    "MRMR": {
        "class": Ranking,
        "config": {
            "eval_method": "mrmr"
        }
    },
}
