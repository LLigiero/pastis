from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.unet_conv_lstm.unet_convlstm import UNetConvLSTMModel

from pastis.ml_logic.models.registry import save_results,load_weights_from_dir


def train_baseline(weights=False,path=''):
    """
    Train baseline model. weights and path are needed if we want to continue an already
    strated training.
    path is a folder in models_output/params/<checkpoint_of_choice>
    PASTIS dates: September 2018 to November 2019
    """
    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-05-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet = Unet_baseline()
    if weights:
        unet.model = load_weights_from_dir(unet.model, path)

    history=unet.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics=history.history

    save_results(metrics)

def train_unet_clstm(weights=False,path=''):

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset()
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm = UNetConvLSTMModel()
    history=unet_clstm.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    if weights:
        unet_clstm.model = load_weights_from_dir(unet_clstm.model, path)

    ##metrics = unet.evaluate_model(pastis.test_dataset)
    metrics=history.history

    save_results(metrics)

if __name__ == '__main__':
    #train_baseline()
    train_unet_clstm()
