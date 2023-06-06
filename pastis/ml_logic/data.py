import tensorflow as tf
from pastis.params import DATA_PATH, TARGET_PATH
import pathlib
import numpy as np
import re


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self) -> None:
        data_path = pathlib.Path(DATA_PATH)
        self.tf_dataset = tf.data.Dataset.list_files(str(data_path / "*"))

        def process_path(file_path):
            path = tf.get_static_value(file_path).decode("utf-8")
            file_name = path.split("/")[-1]
            patch_id = int(re.search(r"_\d*", file_name).group(0)[1:])
            return (
                tf.convert_to_tensor(np.load(path))[0],
                tf.convert_to_tensor(np.load(f"{TARGET_PATH}/TARGET_{patch_id}.npy"))[
                    0
                ],
            )

        self.tf_dataset = self.tf_dataset.map(
            lambda x: tf.py_function(process_path, inp=[x], Tout=[tf.int16, tf.uint8])
        )
