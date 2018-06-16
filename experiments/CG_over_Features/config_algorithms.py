from project.rar.rar import RaR
from project.feature_selection import Filter, SFS, RKNN

ALGORITHMS = {
    "RaR + Deletion": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "deletion",
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 1000,
        }
    },
    "SFS + Tree": {
        "should_impute": False,
        "class": SFS,
        "config": {
            "eval_method": "tree"
        }
    },
    "MI_Filter": {
        "should_impute": False,
        "class": Filter,
        "config": {}
    },
    "RKNN": {
        "should_impute": False,
        "class": RKNN,
        "config": {}
    },
    "RaR + Imputation": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "imputation",
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 1000,
        }
    },
    "RaR + Imputation (Mean)": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "imputation",
            "imputation_method": "simple",
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 1000,
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 1000,
        }
    },
    "KNN + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 1000,
        }
    }
}
