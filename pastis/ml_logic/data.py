import tensorflow as tf
from pastis.params import DATA_PATH
from pastis.ml_logic.utils import process_path, load_geojson
import pathlib
import numpy as np
import re


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, mono_date=True) -> None:
        metadata=load_geojson()

        data_path = pathlib.Path(DATA_PATH)

        self.tf_dataset = tf.data.Dataset.list_files(str(data_path /'DATA_S2'/ "*"))

        self.tf_dataset = self.tf_dataset.map(
            lambda x: tf.py_function(process_path, inp=[x, mono_date], Tout=[tf.float32, tf.uint8])
        )
