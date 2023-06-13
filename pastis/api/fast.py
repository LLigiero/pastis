import pandas as pd
import numpy as np
import ee
import geemap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.registry import load_model_from_name_h5
from pastis.params import *


# run API: uvicorn pastis.api.fast:app --reload
app = FastAPI()

##############################
### loading model from GCP ###
##############################

### Instantiate Model
trained_model = Unet_baseline()
assert trained_model.model is not None

### Load model
name_model='20230612-113429_baseline_aout.h5'
model_load = load_model_from_name_h5(name_model)
weights = model_load.get_weights()
trained_model.model.set_weights(weights)

app.state.model = trained_model.model

##############################
##############################


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# get gps coordonates from front-end
params ={'lat':-1.341984802380476,
         'lon':49.60024167842473}

# call google earth engine api to convert in S2 data

def convert_to_numpy_array(params:dict) -> np.array:
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

    # step 2 : generate image
    #Deffining images to be used for Snetinell (s2) collection
    startDate = '2023-04-01'
    endDate = '2023-06-01'
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate(startDate, endDate)\
        .filterBounds(square).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))\
        .sort('CLOUDY_PIXEL_PERCENTAGE', False)
    #firts image from collection and only 10bands for pastis
    image = s2.first()
    image = image.select(['B2', 'B3', 'B4','B5', 'B6', 'B7','B8','B8A', 'B11', 'B12'])
    #get projection for 10x10m pixel and resample
    proj_10m = image.select('B2').projection()
    img = image.resample('bilinear').reproject(proj_10m)

    # step 3 : convert image to np.array to use for image show and prediction
    image_np = geemap.ee_to_numpy(img, region = square)
    image_np = image_np[:128,:128,:] #assert shape (128,128,10)

    return image_np


# define a root `/` endpoint
@app.get("/")
def root():
    return {'Prediction' : 'I need more pastis'}

# show input image
@app.get("/input_image")
def show_input_image(
    latitude:str,
    longitude:str
    ):
    """
    Plots the test image
    """

    return {'Image_input': 'Image np array'}


# show prediction
@app.get("/predict")
def predict(
    latitude:str,
    longitude:str
    ):
    """
    Make a 128 x 128 image prediction with classes.
    Converts latitute and longitude inputs into S2-type data (shape 128x128x10)
    """

    return {'Baseline_prediction': 'I need more pastis'}
