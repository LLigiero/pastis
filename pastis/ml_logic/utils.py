import os
import json
import ee
import geemap
import numpy as np
import re
import tensorflow as tf
import math
from pastis.params import *
import geopandas as gpd
import pandas as pd

###############################################################################

# GENERAL FUNCTION


def load_geojson() -> gpd:
    """function to load the metadata.geojson, which contain all of the
    needed informations, to synchronize our datas
    """
    metadata = gpd.read_file(META_PATH)
    metadata.index = metadata["ID_PATCH"].astype(int)
    metadata.sort_index(inplace=True)
    return metadata


METADATA = load_geojson()

def rename_file (path_file:str, metrics:str, model_name:str)->str:
    ''' function to rename the outputs_model files, to classify them:
    from : timestamp_
    to :   metrics_modelname_timestamp_'''
    base = path_file.split('/')[:-1]
    base='/'.join(base)+'/'
    end = path_file.split('/')[-1]

    rename = base+metrics+'_'+model_name+'_'+end
    os.rename(path_file, rename)


def index_date(patch_id: int, mono_date: str = '2019-09-01') -> int:
    """Obtain the index of the nearest date of S2 satelite Time Series
    metadata = metadata.geojson (cf load_geojson function)
    patch_id = S2 satelite number
    mono_date = nearest date we want to observe
    """
    mono_date=np.datetime64("2019-09-01")
    date_values = np.array(list(METADATA["dates-S2"][patch_id].values())).astype(str)
    #convert the data to exploit it:
    date_values = np.array(pd.to_datetime(date_values))
    index = np.abs(date_values - mono_date).argmin()
    return index

def pad_time_series_by_zeros(time_series: np.array, size: int) -> np.array:
    """
    Pad time series with np.arrays of zeros in the beggining of the series.
    """
    diff = size - time_series.shape[0]
    if diff < 0:
        raise ValueError("Time series length exceeds expected result length")
    elif diff == 0:
        return time_series

    pads = np.zeros(shape=(diff,) + time_series.shape[1:])
    pad_result = np.concatenate([pads, time_series], axis=0)

    return pad_result

def pad_time_series(time_series: np.array, size: int) -> np.array:
    """Pad the input time series with repeated values without violating the temporal feature of the series.
    The output time series will be of a given length and will contain some elements repeated k times and the rest k+1.
    Example: pad_time_series([1, 2, 3, 4], 10) => [1 1 1 2 2 2 3 3 4 4].
    """
    input_size = time_series.shape[0]
    diff = size - input_size
    if diff < 0:
        return time_series[:TIME_SERIES_LENGTH,:,:,:] #take only the first images
        #raise ValueError("Time series length exceeds expected result length")
    elif diff == 0:
        return time_series

    duplicate_times = math.ceil(size / input_size)
    repeat_times = [duplicate_times] * (size - input_size * (duplicate_times - 1)) + [
        duplicate_times - 1
    ] * (input_size * duplicate_times - size)
    repeat_times[: (size - sum(repeat_times))] = [
        v + 1 for v in repeat_times[: (size - sum(repeat_times))]
    ]

    pad_result = np.repeat(time_series, repeat_times, axis=0)

    return pad_result

def one_hot(y: np.array, num_classes: int) -> np.array:
    """
    One Hot encoder using numpy. Equivalent to to_categorcial from keras.
    from keras.utils.np_utils import to_categorical
    """
    return np.squeeze(np.eye(num_classes)[y])


###############################################################################

# OPTIC SPECIFIC FUNCTIONS

def normalize_patch_spectra(time_series: np.array) -> np.array:
    """Utility function to normalize the Sentinel-2 patch spectra.
    The patch must consist of 10 spectra and the shape n*10*n*n."""

    with open(os.path.join(DATA_PATH, "NORM_S2_patch.json"), "r") as file:
        normvals = json.loads(file.read())
        selected_folds = FOLDS["train"] if FOLDS is not None else range(1, 6)
        means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
        stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
        norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
    result = (time_series - norm[0][None, :, None, None]) / norm[1][None, :, None, None]
    return result

def normalize_image(image: np.array) -> np.array:
    """Utility function to normalize the Sentinel-2 patch spectra.
    The patch must consist of 10 spectra and the shape n*10*n*n."""
    with open(os.path.join(DATA_PATH, "NORM_S2_patch.json"), "r") as file:
        normvals = json.loads(file.read())
        selected_folds = FOLDS["test"] if FOLDS is not None else range(1, 6)
        means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
        stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
        norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
    result = (image - norm[0][None, None, :]) / norm[1][None, None, :]
    return result

def process_path(
    file_path: str,
    mono_date: str,
) -> tf:
    """
    Preprocess time series.
    Output: X, y in tf.tensor for model
    y : semantic classification
    """
    path = tf.get_static_value(file_path).decode("utf-8")
    file_name = path.split("/")[-1]
    patch_id = int(re.search(r"_\d*", file_name).group(0)[1:])

    x = np.load(path)
    x = normalize_patch_spectra(x)

    if mono_date != "0":
        index = index_date(patch_id, mono_date)
        x = x[index].swapaxes(0, 1).swapaxes(1, 2)
        print(patch_id, index)
    else:
        x = pad_time_series(x, TIME_SERIES_LENGTH)
        x = x.swapaxes(1, 3).swapaxes(1, 2)

    target_semantic = np.load(f"{DATA_PATH}/ANNOTATIONS/TARGET_{patch_id}.npy")[0]
    target_semantic_hot = one_hot(target_semantic, NUM_CLASSES)

    x = tf.convert_to_tensor(x.astype(np.float32))
    y = tf.convert_to_tensor(target_semantic_hot.astype(np.uint8))

    return x, y

#####################################################################################

# RADAR and Multimodal functions

def normalize_patch_radar(
    time_series: np.array
    ) -> np.array:
    """Utility function to normalize the Sentinel-1 patch spectra.
    The patch must consist of 3 spectra and the shape n*3*n*n."""

    with open(os.path.join(DATA_PATH, "NORM_S1A_patch.json"), "r") as file:
        normvals = json.loads(file.read())
        selected_folds = FOLDS["train"] if FOLDS is not None else range(1, 6)
        means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
        stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
        norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
    result = (time_series - norm[0][None, :, None, None]) / norm[1][None, :, None, None]
    return result

def process_path_multimodal(
    file_path: str,
    mono_date: str,
    ) -> tf:
    """
    Preprocess time series.
    Output: x-opt, x_radar, y in tf.tensor for model
    """
    path = tf.get_static_value(file_path).decode("utf-8")
    file_name = path.split("/")[-1]
    patch_id = int(re.search(r"_\d*", file_name).group(0)[1:])

    x_radar = np.load(os.path.join(DATA_PATH, "DATA_S1A", f"S1A_{patch_id}.npy"))
    x_radar = normalize_patch_radar(x_radar)

    if mono_date != "0":
        index = index_date(patch_id, mono_date)
        x_radar = x_radar[index].swapaxes(0, 1).swapaxes(1, 2)
    else:
        x_radar = pad_time_series(x_radar, TIME_SERIES_LENGTH)
        x_radar = x_radar.swapaxes(1, 3).swapaxes(1, 2)

    x_opt = np.load(path)
    x_opt = normalize_patch_spectra(x_opt)

    if mono_date != "0":
        index = index_date(patch_id, mono_date)
        x_opt = x_opt[index].swapaxes(0, 1).swapaxes(1, 2)
        print(patch_id, index)
    else:
        x_opt = pad_time_series(x_opt, TIME_SERIES_LENGTH)
        x_opt = x_opt.swapaxes(1, 3).swapaxes(1, 2)

    target_semantic = np.load(f"{DATA_PATH}/ANNOTATIONS/TARGET_{patch_id}.npy")[0]
    target_semantic_hot = one_hot(target_semantic, NUM_CLASSES)

    x = np.concatenate([x_opt,x_radar], axis=-1)
    x = tf.convert_to_tensor(x.astype(np.float32))
    #x_radar = tf.convert_to_tensor(x_radar.astype(np.float32))
    #x_opt = tf.convert_to_tensor(x_opt.astype(np.float32))
    y = tf.convert_to_tensor(target_semantic_hot.astype(np.uint8))

    return x, y



# call google earth engine api to convert in S2 data
def gee_to_numpy_array(params:dict, time_serie=False, startDate='2019-01-01', endDate ='2019-12-31') -> np.ndarray:
    """ Given an input of lat and lon, generates a picture of geometry size similar to our S2 data.
    Return the corresponding np.array
    """

    # connect to earth engine api via service account
    service_account=EARTHENGINE_MAIL
    credentials=ee.ServiceAccountCredentials(service_account, EARTHENGINE_TOKEN)
    ee.Initialize(credentials)

    # step 1 : generates geometry, a square from a point
    point = ee.Geometry.Point([params['lat'], params['lon']])
    areaM2 = 1280*1280
    square = point.buffer(ee.Number(areaM2).sqrt().divide(2), 1).bounds()

    # step 2 : generate image from COPERNICUS/S2
    bands=['B2', 'B3', 'B4','B5', 'B6', 'B7','B8','B8A', 'B11', 'B12']

    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate(startDate, endDate)\
        .filterBounds(square).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))\
        .select(bands)

    if not time_serie:
        # select first image from collection and only 10bands for pastis
        image = s2.first()
        # Get the number of images in the collection
        size = image.size().getInfo()
        print(f'Number of images in the collection : {size}')

        #get projection for 10x10m pixel and resample
        proj_10m = image.select('B2').projection()
        img = image.resample('bilinear').reproject(proj_10m)

        # step 3 : convert image to np.array to use for image show and prediction
        image_np = geemap.ee_to_numpy(img, region = square)
        image_np = image_np[:128,:128,:] #assert shape (128,128,10)

        return image_np

    else:
        # Get the number of images in the collection
        size = s2.size().getInfo()
        print(f'Number of images in the collection : {size}')

        #get projection for 10x10m pixel and map
        proj_10m = s2.first().select('B2').projection()
        s2 = s2.map(lambda image : image.resample('bilinear').reproject(proj_10m))
        #s2 = s2.map(lambda image : image.clip(square.buffer(50)))

        # iterate through image list to reshape in proper format (size x 128 x128 x10)
        s2_list = s2.toList(s2.size())
        list_img = []

        for i in range(size):
            img = ee.Image(s2_list.get(i))
            image_np = geemap.ee_to_numpy(img, region=square)
            image_np = image_np[:128,:128,:]
            list_img.append(image_np)

        return np.array(list_img)
