from project.rar.rar import RaR

ALGORITHMS = {
    "RaR Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
        }
    },
    "RaR Partial": {
        "class": RaR,
        "config": {
            "approach": "partial",
            "n_targets": 0,
        }
    },
    "RaR Fuzzy alpha": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "alpha",
            "n_targets": 0,
        }
    },
    "RaR Fuzzy imputed": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "n_targets": 0,
        }
    },
    "RaR Fuzzy proba": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "proba",
            "n_targets": 0,
        }
    },
    "MEAN + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "approach": "deletion",
            "n_targets": 0,
        }
    },
}
