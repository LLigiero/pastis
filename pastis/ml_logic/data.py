import tensorflow as tf
from pastis.params import DATA_PATH, FOLDS
from pastis.ml_logic.utils import process_path, load_geojson
import os


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, split: str, mono_date: int = 0) -> None:
        assert split in ["train", "val", "test"]

        metadata = load_geojson()

        paths = metadata[metadata.Fold.isin(FOLDS[split])]
        paths = paths.ID_PATCH.map(
            lambda id: str(os.path.join(DATA_PATH, "DATA_S2", f"S2_{id}.npy"))
        )
        paths = list(paths)

        self.tf_dataset = tf.data.Dataset.list_files(paths)

        self.tf_dataset = self.tf_dataset.map(
            lambda path: tf.py_function(
                process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8]
            )
        )
