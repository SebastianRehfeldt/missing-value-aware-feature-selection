from project.rar.rar import RaR

SHARED = {
    "n_targets": 0,
    "boost_value": 0,
    "boost_inter": 0,
    "boost_corr": 0,
    "n_subspaces": 500,
    "subspace_size": (2, 2),
    "active_sampling": False,
}

ALGORITHMS = {
    "RaR Deletion": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "RaR Deletion Category": {
        "class": RaR,
        "config": {
            "approach": "deletion",
            "create_category": True,
            **SHARED
        }
    },
    "RaR Partial": {
        "class": RaR,
        "config": {
            "approach": "partial",
            **SHARED
        }
    },
    "RaR Fuzzy alpha": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "alpha",
            **SHARED
        }
    },
    "RaR Fuzzy distance": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "imputation_method": "soft",
            "dist_method": "distance",
            **SHARED
        }
    },
    "RaR Fuzzy radius": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "imputed",
            "imputation_method": "soft",
            "dist_method": "radius",
            **SHARED
        }
    },
    "RaR Fuzzy proba": {
        "class": RaR,
        "config": {
            "approach": "fuzzy",
            "weight_approach": "proba",
            **SHARED
        }
    },
    "MEAN + RaR": {
        "should_impute": True,
        "strategy": "simple",
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "MICE + RaR": {
        "should_impute": True,
        "strategy": "mice",
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
    "DELETION + RaR": {
        "should_delete": True,
        "class": RaR,
        "config": {
            "approach": "deletion",
            **SHARED
        }
    },
}
