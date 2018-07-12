CONFIG = {
    "n_runs": 2,
    "n_insertions": 2,  # maximum 10 insertions
    "seeds": [42, 0, 13, 84, 107, 15, 23, 11, 174, 147],
    "missing_rates": [0.0, 0.3],  # , 0.5, 0.7, 0.8
    "is_real_data": False,
    "update_config": False,
    "updates": [
        {
            "n_samples": 500
        },
        {
            "n_samples": 1000
        },
    ],
}
