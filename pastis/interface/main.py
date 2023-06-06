from pastis.ml_logic.utils import load_geojson

from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.baseline_model import baseline_unet_model, compile_model, train_model
from sklearn.model_selection import train_test_split


def init_train_baseline_model():

    # Instantiate class instance
    pastis = PastisDataset()

    # ----- TRAIN TEST SPLIT -----
    '''Assuming data already preprocessed'''
    # '''Create ds_train, ds_val, ds_test)'''

    # ----- TRAIN MODEL -----
    '''Define model using `baseline_model.py`'''
    model = baseline_unet_model(dropout=0.2)

    '''Compile model using `baseline_model.py`'''
    model = compile_model(model, learning_rate=0.05)

    '''Train model using `baseline_model.py`'''
    batch_size = 28
    patience = 2

    model, history = train_model(
            model,
            pastis.tf_dataset,
            batch_size=batch_size,
            patience=patience,
        )

    return model, history

if __name__ == '__main__':
    METADATA=load_geojson()
