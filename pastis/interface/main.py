from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.unet_conv_lstm.unet_convlstm import UNetConvLSTMModel

from pastis.ml_logic.models.registry import save_results, load_model_from_name_h5, save_model


def train_baseline(saved_model=False,name_model=''):
    """
    Train baseline model. saved_model=True  and name_model (str) are need to
    load model if saved.
    name_model is a file name in models_output/model_h5/<file_name>
    PASTIS dates: September 2018 to November 2019
    """
    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-05-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet = Unet_baseline()

    if saved_model:
        model_load = load_model_from_name_h5(name_model)
        weights = model_load.get_weights()
        unet.model.set_weights(weights)

    unet.model, history = unet.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics = history.history
    save_model(unet.model)
    save_results(metrics)

def train_unet_clstm(saved_model=False,name_model=''):

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset()
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm = UNetConvLSTMModel()

    if saved_model:
        model_load = load_model_from_name_h5(name_model)
        weights = model_load.get_weights()
        unet_clstm.model.set_weights(weights)

    unet_clstm.model, history = unet_clstm.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics=history.history
    save_model(unet_clstm.model)
    save_results(metrics)


def evaluate_unet_clstm(name_model=''):

    # Instantiate class instance
    print("Initial data")
    pastis = PastisDataset()
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm = UNetConvLSTMModel()
    assert unet_clstm.model is not None

    #load model
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    unet_clstm.model.set_weights(weights)

    metrics = unet_clstm.evaluate_model(pastis.test_dataset)
    save_results(metrics)


if __name__ == '__main__':
    train_baseline()
    train_unet_clstm()
    evaluate_unet_clstm()
