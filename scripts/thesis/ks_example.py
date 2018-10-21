# %%
import numpy as np
import pandas as pd

feature = [1, 1, 2, 3, 3, 4, 5, 6]
marginal = [1, 1, 1, 1, 1, 1, 1, 1]
mass_marginal = [(i + 1) / 8 for i in range(8)]

slice1 = [1, 1, 0, 0, 1, 0, 0, 0]
mass_slice1 = [1 / 3, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1, 1]
slice2 = [0, 0, 0, 1, 0, 1, 0, 1]
mass_slice2 = [0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1]

is_last = ["no", "yes", "yes", "no", "yes", "yes", "yes", "yes"]

shuffle = np.random.permutation(len(feature))

df = pd.DataFrame({
    "feature values": feature,
    "marginal": marginal,
    "mass marginal": mass_marginal,
    "slice1": slice1,
    "mass slice1": np.round(mass_slice1, 3),
    "slice2": slice2,
    "mass slice2": np.round(mass_slice2, 3),
    "is_last": is_last
}).T
df

# %%
print(df.to_latex())

# %%
df = df.T
df["dist1"] = np.abs(df["mass slice1"] - df["mass marginal"])
df["dist2"] = np.abs(df["mass slice2"] - df["mass marginal"])

# %%
print(df.T.to_latex())
