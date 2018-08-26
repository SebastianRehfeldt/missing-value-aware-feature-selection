CONFIG = {
    "n_runs": 5,
    "n_insertions": 5,  # maximum 10 insertions
    "seeds": [42, 0, 13, 84, 107, 15, 23, 11, 174, 147],
    "missing_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "is_real_data": False,
    "update_config": False,  # update params must be >= n_runs
    "update_attribute": "n_samples",
    "updates": [
        {
            "n_samples": 500
        },
        {
            "n_samples": 2000
        },
    ],
}
