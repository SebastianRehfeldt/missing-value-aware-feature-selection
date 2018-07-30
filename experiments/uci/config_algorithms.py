from project.rar.rar import RaR

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
