import os
import json
import numpy as np
import re
import tensorflow as tf
from pastis.params import *

def normalize_patch_spectra(time_series):
    """Utility function to normalize the Sentinel-2 patch spectra.
       The patch must consist of 10 spectra and the shape n*10*n*n."""

    with open (os.path.join(DATA_PATH, "NORM_S2_patch.json"), "r") as file:
            normvals = json.loads(file.read())
            selected_folds = FOLDS if FOLDS is not None else range(1, 6)
            means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
            stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
            norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
    result = (time_series - norm[0][None, :, None, None])/ norm[1][None, :, None, None]
    return result.astype(np.float32)

def pad_time_series_by_zeros(time_series, size):
    diff = size - time_series.shape[0]
    if diff < 0:
        raise ValueError("Time series length exceeds expected result length")
    elif diff == 0:
        return time_series

    pads = np.zeros(shape=(diff,) + time_series.shape[1:])
    pad_result = np.concatenate([pads, time_series], axis=0)

    return pad_result

def process_path(file_path):
    path = tf.get_static_value(file_path).decode("utf-8")
    file_name = path.split("/")[-1]
    patch_id = int(re.search(r"_\d*", file_name).group(0)[1:])
    x = np.load(path)
    x = normalize_patch_spectra(x)

    return (
        tf.convert_to_tensor(x)[0],
        tf.convert_to_tensor(np.load(f"{DATA_PATH}/ANNOTATIONS/TARGET_{patch_id}.npy"))[0],
            )
