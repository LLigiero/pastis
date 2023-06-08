from pastis.ml_logic.data import PastisDataset
from pastis.ml_logic.models.baseline_model import Unet_baseline
from pastis.ml_logic.models.registry import save_results

def train_baseline():

    # Instantiate class instance

    print("Initial data")
    pastis = PastisDataset('2019-08-16')
    print("tfds object is ready ")

    # Instantiate Model
    unet = Unet_baseline()
    history =unet.fit_model(pastis.train_dataset, validation_ds=pastis.val_dataset)

    metrics = unet.evaluate_model(pastis.test_dataset)
    params=history.history

    save_results(params)

if __name__ == '__main__':
    train_baseline()
