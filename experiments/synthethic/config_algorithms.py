from project.rar.rar import RaR
from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange
from project.feature_selection.ranking import Ranking
from project.feature_selection.embedded import Embedded
from project.feature_selection.baseline import Baseline

ALGORITHMS = {
    "Baseline": {
        "class": Baseline,
        "config": {}
    },
    "FCBF SK": {
        "class": Ranking,
        "config": {
            "eval_method": "fcbf"
        }
    },
    "RaR Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
        }
    },
}
"""
    "RaR Fuzzy": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "n_targets": 0,
        }
    },
    "XGBoost": {
        "class": Embedded,
        "config": {}
    },
    "Relief Partial": {
        "class": Ranking,
        "config": {
            "eval_method": "myrelief"
        }
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
"""
