from pastis.ml_logic.utils import normalize_patch_spectra


def predict_model(X_pred, model):
    """
    x=numpy array with correct shape

    Make a prediction using the latest trained model
    """
    #model = load_model()
    assert model is not None

    X_processed= normalize_patch_spectra(X_pred)
    y_pred = model.predict(X_processed)

    return y_pred
