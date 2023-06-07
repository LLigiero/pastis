import tensorflow as tf
from pastis.params import DATA_PATH, FOLDS
from pastis.ml_logic.utils import process_path, load_geojson
import os


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, mono_date: int = 0) -> None:
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
        self.test_dataset = tf.data.Dataset.list_files(files_list["val"])
        self.test_dataset = self.test_dataset.map(
            lambda path: tf.py_function(
                process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8]
            )
        )
