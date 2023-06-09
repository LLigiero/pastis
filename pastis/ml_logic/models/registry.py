import os
import time
import pickle
from keras.models import Model
from pastis.params import SAVE_PATH


def save_results(metrics: dict) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    # if params is not None:
    #     params_path = os.path.join(SAVE_PATH, "params", timestamp + ".pickle")
    #     with open(params_path, "wb") as file:
    #         pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(SAVE_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def load_weights_from_dir(model: Model, path:str) -> Model:
    """
    Load weights saved in path into initialied model. Can be used to continue trainning,
    evaluate/pred.
    path is a folder in models_output/params/<checkpoint_of_choice>
    example: path='./models_output/params/20230609-100711modcheck'
    """
    print("✅ Weights loaded to model")
    return model.load_weights(path)
