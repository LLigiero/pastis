import pandas as pd
import numpy as np
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pastis.ml_logic.models.unet_baseline.baseline_model import Unet_baseline
from pastis.ml_logic.models.unet_conv_lstm.unet_convlstm import UNetConvLSTMModel
from pastis.ml_logic.models.registry import load_model_from_name_h5
from pastis.ml_logic.utils import gee_to_numpy_array
from pastis.interface.main import predict_model_unet, predict_model_unet_clstm
from pastis.params import *
from contextlib import asynccontextmanager
from pastis.api.test import res

model={}

@asynccontextmanager
async def lifespan(app):
    # Unet_baseline
    name_model='20230612-113429.h5'
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    trained_model = Unet_baseline()
    trained_model.model.set_weights(weights)
    model['unet'] = trained_model

    # Unet_convlstm
    name_model='20230613-065205_unet_convlstm_suite.h5'
    model_load = load_model_from_name_h5(name_model)
    weights = model_load.get_weights()
    trained_model = UNetConvLSTMModel(NUM_CLASSES)
    trained_model.model.set_weights(weights)
    model['clstm'] = trained_model

    yield
    model.clear()

# run API: uvicorn pastis.api.fast:app --reload
# app = FastAPI()
app = FastAPI(lifespan=lifespan)


##############################
### loading model from GCP ###
##############################

### Instantiate Model
# assert trained_model.model is not None

### Load model
 # TO DO: get model from bucket

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
        _out = predict_model_unet(_in, name_model, model['unet'])
    else:
        name_model='20230613-065205_unet_convlstm_suite.h5'
        _out = predict_model_unet_clstm(_in, name_model, model['clstm'])

    print ("Send data DONE")
    return {'patch': json.dumps(_in.tolist()),
            'pred' : json.dumps(_out.tolist())}

    # return {'patch': json.dumps(_in.tolist()),
    #         'pred' : json.dumps(_out.tolist())}

@app.get("/predict_test")
async def test_front():
    return {'pred' : res}
