import tensorflow as tf
from pastis.params import DATA_PATH, FOLDS
from pastis.ml_logic.utils import process_path, load_geojson
from pastis.ml_logic.utils import process_path_multimodal
import os


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, mono_date: str = "0") -> None:
        ''' For selecting only one date of the time series, instantiate the class
        with a mono_date format like "YYYY-MM-DD" '''

        metadata = load_geojson()

        splits = ["train", "val", "test"]
        files_list = {}

        for split in splits:
            paths = metadata[metadata.Fold.isin(FOLDS[split])]
            paths = paths.ID_PATCH.map(
                lambda id: str(os.path.join(DATA_PATH, "DATA_S2", f"S2_{id}.npy"))
            )
            files_list[split] = list(paths)

        self.train_dataset = tf.data.Dataset.list_files(files_list["train"])
        self.train_dataset = self.train_dataset.map(
            lambda path: tf.py_function(
                process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8]
            )
        )
        self.val_dataset = tf.data.Dataset.list_files(files_list["val"])
        self.val_dataset = self.val_dataset.map(
            lambda path: tf.py_function(
                process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8]
            )
        )
        self.test_dataset = tf.data.Dataset.list_files(files_list["test"])
        self.test_dataset = self.test_dataset.map(
            lambda path: tf.py_function(
                process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8]
            )
        )

class PastisDataset_Multimodal:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, mono_date: str = "0") -> None:
        ''' For selecting only one date of the time series, instantiate the class
        with a mono_date format like "YYYY-MM-DD" '''

        metadata = load_geojson()

        splits = ["train", "val", "test"]

        files_list = {}

        for split in splits:
            paths = metadata[metadata.Fold.isin(FOLDS[split])]
            paths = paths.ID_PATCH.map(
                lambda id: str(os.path.join(DATA_PATH, "DATA_S2", f"S2_{id}.npy"))
            )
            files_list[split] = list(paths)

        self.train_dataset = tf.data.Dataset.list_files(files_list["train"])
        self.train_dataset = self.train_dataset.map(
            lambda path: tf.py_function(
                process_path_multimodal, inp=[path, mono_date], Tout=[tf.float32,tf.float32, tf.uint8]
            )
        )
        self.val_dataset = tf.data.Dataset.list_files(files_list["val"])
        self.val_dataset = self.val_dataset.map(
            lambda path: tf.py_function(
                process_path_multimodal, inp=[path, mono_date], Tout=[tf.float32,tf.float32, tf.uint8]
            )
        )
        self.test_dataset = tf.data.Dataset.list_files(files_list["test"])
        self.test_dataset = self.test_dataset.map(
            lambda path: tf.py_function(
                process_path_multimodal, inp=[path, mono_date], Tout=[tf.float32,tf.float32, tf.uint8]
            )
        )
