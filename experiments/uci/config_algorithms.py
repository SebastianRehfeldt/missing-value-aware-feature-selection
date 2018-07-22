from project.rar.rar import RaR
from project.feature_selection import Filter, SFS, RKNN

ALGORITHMS = {
    "RaR + Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 3,
            "n_subspaces": 800,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Partial": {
        "class": RaR,
        "config": {
            "approach": "partial",
            "n_targets": 3,
            "n_subspaces": 800,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Fuzzy 1": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "n_targets": 3,
            "n_subspaces": 800,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
            "weight": 1,
        }
    },
    "RaR Fuzzy 10": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "n_targets": 3,
            "n_subspaces": 800,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
            "weight": 0.1,
        }
    },
    "RaR Fuzzy 100": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "n_targets": 3,
            "n_subspaces": 800,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
            "weight": 0.01,
        }
    },
}
"""
    "RaR + Imputation (Mean)": {
        "should_impute": False,
        "class": RaR,
        "config": {
            "approach": "imputation",
            "imputation_method": "simple",
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 5000,
        }
    },
    "Mean + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 5000,
        }
    },
    "Deletion + RaR": {
        "should_impute": False,
        "should_delete": True,
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 5000,
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 5000,
        }
    },
    "KNN + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "use_pearson": False,
            "n_targets": 0,
            "n_subspaces": 5000,
        }
    },
    "SFS + Tree": {
        "should_impute": False,
        "class": SFS,
        "config": {
            "eval_method": "tree"
        }
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
    "MI_Filter": {
        "should_impute": False,
        "class": Filter,
        "config": {}
    },
"""
