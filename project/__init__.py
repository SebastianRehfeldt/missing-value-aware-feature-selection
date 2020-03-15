import os

__version__ = "0.1.0"

PROJECT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))
ROOT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, os.pardir))
DATA_PATH = os.path.join(ROOT_PATH, "data")

EXPERIMENTS_PATH = os.path.join(ROOT_PATH, "thesis-experiments")
