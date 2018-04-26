import os
import pandas as pd
from project import DATA_PATH

def load_csv(name="test", header=None):
    url = os.path.join(DATA_PATH, name) 
    df = pd.read_csv(url + ".csv", header=header)
    return df