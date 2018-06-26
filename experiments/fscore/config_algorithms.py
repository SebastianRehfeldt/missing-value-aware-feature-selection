from project.rar.rar import RaR
from project.feature_selection import Filter, SFS, RKNN

ALGORITHMS = {
    "RaR + Deletion": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "deletion",
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 2000,
            "contrast_iterations": 250,
        }
    },
    "Mean + RaR + Pearson": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "use_pearson": True,
            "n_targets": 3,
            "n_subspaces": 2000,
            "contrast_iterations": 250,
        }
    },
    "Mean + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 2000,
            "contrast_iterations": 250,
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 2000,
            "contrast_iterations": 250,
        }
    },
    "KNN + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 2000,
            "contrast_iterations": 250,
        }
    },
    "RKNN": {
        "should_impute": False,
        "class": RKNN,
        "config": {
            # "n_subspaces": 500,
        },
    },
    "MI_Filter": {
        "should_impute": False,
        "class": Filter,
        "config": {}
    },
}
"""
    "Deletion + RaR": {
        "should_impute": False,
        "should_delete": True,
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 2000,
        }
    },
    "SFS + Tree": {
        "should_impute": False,
        "class": SFS,
        "config": {
            "eval_method": "tree"
        }
    },
    "RaR + Imputation": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "imputation",
            "use_pearson": False,
            "n_targets": 3,
            "n_subspaces": 1000,
        }
    },
"""
