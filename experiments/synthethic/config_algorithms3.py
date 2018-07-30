from project.rar.rar import RaR
from project.feature_selection import RKNN, Filter, SFS
from project.feature_selection.orange import Orange

ALGORITHMS = {
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
    "RaR Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Partial": {
        "class": RaR,
        "config": {
            "approach": "partial",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Fuzzy": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "MEAN + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (1, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
}
