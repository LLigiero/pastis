import pandas as pd
import numpy as np
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.registry import load_model_from_name_h5
from pastis.ml_logic.utils import gee_to_numpy_array
from pastis.interface.main import predict_model_unet, predict_model_unet_clstm
from pastis.params import *


# run API: uvicorn pastis.api.fast:app --reload
app = FastAPI()

##############################
### loading model from GCP ###
##############################

### Instantiate Model
trained_model = Unet_baseline()
# assert trained_model.model is not None

### Load model
name_model='20230612-113429.h5' # TO DO: get model from bucket
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

# define a root `/` endpoint
@app.get("/")
def root():
    return {'Prediction' : 'Give me more pastis, bro'}

# return input and output image of type np.ndarray
@app.get("/predict")
def get_input_output_image(
    latitude:str,
    longitude:str,
    time_serie:bool,
    start_date:str,
    end_date:str,
    ):
    """
    Returns the input and predicted images in np.ndarray format
    """
    params ={
        'lat':float(latitude),
        'lon':float(longitude)
        }

    _in = gee_to_numpy_array(params, time_serie, start_date, end_date)

    if not time_serie :
        name_model='20230612-113429.h5'
        _out = predict_model_unet(_in, name_model)
    else:
        name_model='20230613-065205_unet_convlstm_suite.h5'
        _out = predict_model_unet_clstm(_in, name_model)

    return {'patch': json.dumps(_in.tolist()),
            'pred' : json.dumps(_out.tolist())}

    # return {'patch': json.dumps(_in.tolist()),
    #         'pred' : json.dumps(_out.tolist())}
