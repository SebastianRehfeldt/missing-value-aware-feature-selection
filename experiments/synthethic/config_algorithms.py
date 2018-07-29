from project.rar.rar import RaR

ALGORITHMS = {
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
    "RaR Deletion 2": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (2, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Partial 2": {
        "class": RaR,
        "config": {
            "approach": "partial",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (2, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR Fuzzy 2": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "n_targets": 0,
            "n_subspaces": 600,
            "subspace_size": (2, 3),
            "alpha": 0.02,
            "contrast_iterations": 250,
        }
    },
    "RaR + KNN": {
        "class": RaR,
        "config": {
            "approach": "imputation",
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
    "KNN + RaR": {
        "should_impute": True,
        "strategy": "knn",
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
    "SOFT + RaR": {
        "should_impute": True,
        "strategy": "soft",
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
    "Deletion + RaR": {
        "should_delete": True,
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
