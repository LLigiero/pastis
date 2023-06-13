import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.registry import load_model_from_name_h5
import numpy as np

# run API: uvicorn pastis.api.fast:app --reload
app = FastAPI()

##############################
### loading model from GCP ###
##############################

### Instantiate Model
#trained_model = Unet_baseline()
#assert trained_model.model is not None

### Load model
#name_model='20230612-113429.h5'
#model_load = load_model_from_name_h5(name_model)
#weights = model_load.get_weights()
#trained_model.model.set_weights(weights)

# app.state.model = load_model_from_name_h5(trained_model)

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

def convert_to_numpy_array(params:dict) -> np.ndarray:
    """ Given an input of lat and lon, generates a picture of geometry size similar to our S2 data.
    Return the corresponding np.array
    """

    # step 1 : generates geometry
    # example: gemoetry = [
        # [[-1.270332050595563, 49.6353097251849],
        #  [-1.2526136350202335, 49.635043589013904],
        #  [-1.2530253544991803, 49.623535818998874],
        #  [-1.270739600356701, 49.62380184758293],
        #  [-1.270332050595563, 49.6353097251849]]
        # ]

    # step 2 : generate image


    # step 3 : convert image to np.array to use for image show and prediction

    pass


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
