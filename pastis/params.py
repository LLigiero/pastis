import os
from pastis.ml_logic.utils import load_geojson

DATA_PATH = os.getenv("DATA_PATH")
TARGET_PATH = os.getenv("TARGET_PATH")

#Metadata path
META_PATH = os.getenv("META_PATH")

TIME_SERIES_LENGTH = 70
FOLDS = None
