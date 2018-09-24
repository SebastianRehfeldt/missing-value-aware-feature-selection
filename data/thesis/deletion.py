# %%
import numpy as np
import pandas as pd

k_s = [1,2,3,20]
mr_s = [0, 0.05] + [0.1 * i for i in range(1, 10)]
mr_s = np.round(mr_s, 2)
n = 500

df = pd.DataFrame(np.ones((len(k_s), len(mr_s))), index=k_s, columns=mr_s)

for k in k_s:
    for mr in mr_s:
        df.loc[k,mr] = (1-mr)**k * n

df = df.round(1)
print(df.to_latex())
