from pastis.ml_logic.utils import load_geojson

from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.models.baseline_model import Unet_baseline
from sklearn.model_selection import train_test_split


def train_baseline(model="baseline"):

    # Instantiate class instance
    pastis = PastisDataset('2019-08-16')

    #
    unet = Unet_baseline()
    unet.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)
    unet.evaluate_model(pastis.test_dataset)

if __name__ == '__main__':
    METADATA=load_geojson()
