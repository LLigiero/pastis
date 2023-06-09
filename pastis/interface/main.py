import numpy as np
import tensorflow as tf

from pastis.ml_logic.data import PastisDataset, PastisDataset_Multimodal
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.unet_conv_lstm.unet_convlstm import UNetConvLSTMModel
from pastis.ml_logic.models.unet_radar.unet_convlstm_radar import UNetConvLSTMModel_Multimodal
from pastis.ml_logic.models.registry import save_results, load_model_from_name_h5, save_model
from pastis.ml_logic.models.results_viz import plot_history
from pastis.ml_logic.utils import normalize_image, normalize_patch_spectra, pad_time_series
from pastis.params import *


def train_baseline(saved_model=False,name_model=''):
    """
    Train baseline model. saved_model=True  and name_model (str) are need to
    load model if saved.
    name_model is a file name in models_output/model_h5/<file_name>
    PASTIS dates: September 2018 to November 2019
    """
    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-08-16')
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

def evaluate_unet(name_model='20230612-113429_baseline_aout.h5'):

    # Instantiate class instance
    print("Initial data")
    pastis = PastisDataset('2019-08-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet = Unet_baseline()
    assert unet.model is not None

    #load model
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    unet.model.set_weights(weights)

    metrics = unet.evaluate_model(pastis.test_dataset)
    save_results(metrics)


def train_unet_clstm(saved_model=True,name_model='20230613-065205_unet_convlstm_suite.h5'):

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


def train_unet_clstm_radar(saved_model=False,name_model=''):

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset_Multimodal()
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm_radar = UNetConvLSTMModel_Multimodal()

    if saved_model:
        model_load = load_model_from_name_h5(name_model)
        weights = model_load.get_weights()
        unet_clstm_radar.model.set_weights(weights)

    unet_clstm_radar.model, history = unet_clstm_radar.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics=history.history
    save_model(unet_clstm_radar.model)
    save_results(metrics)


def evaluate_unet_clstm(name_model='20230613-065205_unet_convlstm_suite.h5'):

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

def evaluate_unet_clstm_radar(name_model='20230615-113335_unet_convlstm_multimodal.h5'):

    # Instantiate class instance
    print("Initial data")
    pastis = PastisDataset_Multimodal()
    print("tfds object is ready ")

    # Instantiate Model
    unet_clstm_multi = UNetConvLSTMModel_Multimodal()
    assert unet_clstm_multi.model is not None

    #load model
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    unet_clstm_multi.model.set_weights(weights)

    metrics = unet_clstm_multi.evaluate_model(pastis.test_dataset)
    save_results(metrics)


def predict_model_unet(X_new,name_model='20230612-113429_baseline_aout.h5'):
    """
    X_new=numpy array with correct shape(128,128,10) or (10,128,128)
    Make a prediction using the latest Unet trained model
    Output: np.array (128,128) if ok; str if error
    """
    # Instantiate Model
    unet_baseline = Unet_baseline()
    #unet_clstm = UNetConvLSTMModel()


    #load model
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    unet_baseline.model.set_weights(weights)

    assert unet_baseline.model is not None

    if X_new.shape == (128,128,10):
        X_new_processed= normalize_image(X_new)
        X_new_processed = np.expand_dims(X_new_processed, axis=0)
        y_pred = unet_baseline.model.predict(X_new_processed)
        y_pred = np.argmax(y_pred[0], axis=2)
        return y_pred

    if X_new.shape == (10,128,128):
        X_new = X_new.swapaxes(0, 1).swapaxes(1, 2)
        X_new_processed= normalize_image(X_new)
        X_new_processed = np.expand_dims(X_new_processed, axis=0)
        y_pred = unet_baseline.model.predict(X_new_processed)
        y_pred = np.argmax(y_pred[0], axis=2)
        return y_pred

    return 'Please verify input shape'



def predict_model_unet_clstm(X_new,name_model='20230613-065205_unet_convlstm_suite.h5'):
    """
    TIME SERIES LENGH = 61 for this model
    X_new=numpy array with correct shape(:,10, 128,128) or (:,128,128,10)
    Make a prediction using the latest Unet trained model
    Output: np.array (128,128) if ok; str if error
    """
    # Instantiate Model
    unet_clstm = UNetConvLSTMModel()

    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    unet_clstm.model.set_weights(weights)
    assert unet_clstm.model is not None

    if X_new.shape[1:] == (128,128,10):
        X_new = X_new.swapaxes(1, 3).swapaxes(2, 3)
        X_new_processed= normalize_patch_spectra(X_new)
        X_new_processed = pad_time_series(X_new_processed, TIME_SERIES_LENGTH)
        X_new_processed = X_new_processed.swapaxes(1, 3).swapaxes(1, 2)
        X_new_processed = np.expand_dims(X_new_processed, axis=0)
        y_pred = unet_clstm.model.predict(X_new_processed)
        y_pred = np.argmax(y_pred[0], axis=2)
        return y_pred

    if X_new.shape[1:] == (10,128,128):
        X_new_processed= normalize_patch_spectra(X_new)
        X_new_processed = pad_time_series(X_new_processed, TIME_SERIES_LENGTH)
        X_new_processed = X_new_processed.swapaxes(1, 3).swapaxes(1, 2)
        X_new_processed = np.expand_dims(X_new_processed, axis=0)
        y_pred = unet_clstm.model.predict(X_new_processed)
        y_pred = np.argmax(y_pred[0], axis=2)
        return y_pred

    return 'Please verify input shape'



if __name__ == '__main__':
    #train_baseline()
    #evaluate_unet()
    #predict_model_unet()
    #train_unet_clstm()
    evaluate_unet_clstm()
    #predict_model_unet_clstm()
    #train_unet_clstm_radar()
