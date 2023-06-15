import os
import io
import time
import pickle
import glob
from keras.models import Model, load_model
from pastis.params import SAVE_PATH, NUM_CLASSES, MODEL_TARGET, ROOT_BUCKET
from pastis.ml_logic.models.metrics import _iou, m_iou
from colorama import Fore, Style
from google.cloud import storage

import subprocess

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
    - locally (latest one in alphabetical order) if MODEL_TARGET=='local'
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(SAVE_PATH, 'model_h5', model_name)
        model = load_model(local_model_directory,
                        custom_objects={'iou':_iou(NUM_CLASSES).iou,
                                        'mean_iou':m_iou(NUM_CLASSES).mean_iou})

        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        print("✅ Model loaded from local disk")

    elif MODEL_TARGET == "gcs":

        # checking if model_to_load is baseline or conv_lstm and edit path to proper folder
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        bucket = client.get_bucket(ROOT_BUCKET)
        blobs = bucket.list_blobs(prefix='Models/2023')

        #try:
        for blob in blobs:
            print('-'*50)
            print(blob.name)

            # model_path_to_save = os.path.join('models_output', 'model_h5', f'{name}_{timestamp}.h5')
            # print(model_path_to_save)

            file = io.BytesIO()

            if model_name in blob.name:
                print(blob.name)
                blob.download_to_filename('cache')
                model = load_model("./cache",
                                custom_objects={'iou':_iou(NUM_CLASSES).iou,
                                                'mean_iou':m_iou(NUM_CLASSES).mean_iou})

                print('-'*50)
                print(f'✅ Model {blob.name} downloaded from cloud storage')
                return model


        # except:
        #     print(f"\n❌ No model found in GCS bucket {bucket}")
        #     return None

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
