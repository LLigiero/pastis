import tensorflow as tf
from pastis.params import DATA_PATH
from pastis.ml_logic.utils import process_path
import pathlib


class PastisDataset:
    """
    Custom Class wrapper to manipulate Tf dataset with S2 data
    """

    def __init__(self, mono_date:int=0) -> None:

        # self.metadata=load_geojson()
        # id_patch_list = list(self.metadata['ID_PATCH'])
        # dates_s2_list = list(self.metadata['dates-S2'].apply(lambda x: list(x.values())))
        # print (type(dates_s2_list))
        # print (dates_s2_list)


        data_path = pathlib.Path(DATA_PATH)


        self.tf_dataset = tf.data.Dataset.list_files(str(data_path /'DATA_S2'/ "*"))

        self.tf_dataset = self.tf_dataset.map(
            lambda path: tf.py_function(process_path, inp=[path, mono_date], Tout=[tf.float32, tf.uint8])
        )
