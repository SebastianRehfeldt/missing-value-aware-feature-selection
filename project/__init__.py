import os
from collections import namedtuple
from project.utils.data import Data

PROJECT_PATH = os.path.abspath(os.path.join(
    os.path.abspath(__file__), os.pardir))
ROOT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, os.pardir))
DATA_PATH = os.path.join(ROOT_PATH, "data")
