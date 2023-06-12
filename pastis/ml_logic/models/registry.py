import os
import time
import pickle
from keras.models import Model, load_model
from pastis.params import SAVE_PATH, NUM_CLASSES
from pastis.ml_logic.models.metrics import _iou, m_iou


def save_model(model:Model = None) -> None:
    """
    Persist trained model locally on the hard drive at
    f"{SAVE_PATH}/model_h5/{timestamp}.h5"

    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(SAVE_PATH, 'model_h5', f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None

def load_model_from_name_h5(model_name:str) -> Model:
    """
    Return a saved model:
    """
    model_directory = os.path.join(SAVE_PATH, 'model_h5', model_name)
    model = load_model(model_directory,
                        custom_objects={'iou':_iou(NUM_CLASSES).iou,
                                        'mean_iou':m_iou(NUM_CLASSES).mean_iou})

    print("✅ Model loaded from local disk")

    return model

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
    DO NOT WORK!!
    Load weights saved in path into initialied model. Can be used to continue trainning,
    evaluate/pred.
    path is a folder in models_output/params/<checkpoint_of_choice>
    example: path='./models_output/params/20230609-100711modcheck'
    """
    print("✅ Weights loaded to model")
    return model.load_weights(path)
