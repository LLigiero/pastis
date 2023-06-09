import os
import time

from google.cloud import storage
from tensorflow import keras

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".pastis", "pastis_outputs")
print(LOCAL_REGISTRY_PATH)

MODEL_TARGET='gcs'
BUCKET_NAME='models'

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in GCS bucket at "models/{timestamp}.h5" --> unit 02 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, 'models', f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None
