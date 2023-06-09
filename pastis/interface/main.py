from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.unet_conv_lstm.unet_convlstm import UNetConvLSTMModel

from pastis.ml_logic.models.registry import save_results

def train_baseline():

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-08-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet = Unet_baseline()
    history=unet.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics=history.history

    save_results(metrics)

def train_unet_clstm():

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-08-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm = UNetConvLSTMModel()
    history=unet_clstm.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics=history.history

    save_results(metrics)

if __name__ == '__main__':
    #train_baseline()
    train_unet_clstm()
