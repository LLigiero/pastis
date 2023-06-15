from streamlit_folium import st_folium
import streamlit as st
from pastis.streamlit.app_utils import *
import numpy as np
import json

import folium

app_init()

user_pickup_sat_temp = st.text_input('Adress / Coordinates',DEFAULT_COORDINATES)
col1, col2,col3 = st.columns(3)
predict_button=col2.button('click me for prediction',)


##### Init st.session_state
if "center" not in st.session_state:
    st.session_state["center"] = [47.78530809282205 , 2.6266160414864257]
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 6
if "markers" not in st.session_state:
    st.session_state["markers"] = []
#####


if predict_button:
    lon, lat= get_coordinates(user_pickup_sat_temp)
    # Generate the polygon where the target will be display on map
    polygon=get_square_dict((float(lat),float(lon)))

    #url = 'http://127.0.0.1:8000/predict_test'
    # predict = requests.get(url).json()
    # target = np.array(predict['pred']) # y_pred
    url = 'http://127.0.0.1:8000/predict?'
    params={
    'latitude': lat , 'longitude': lon,
    'time_serie': True,
    'start_date': '2018-05-01',
    'end_date': '2019-09-01'}

    predict = requests.get(url,params).json()
    target = np.array(json.loads(predict['pred'])) # y_pred
    print(f'target_shape={target.shape}')

    # marker = folium.Marker(
    #     location=[float(lon),float(lat)],
    #     popup=f"Random marker at {float(lat):.2f}, {float(lon):.2f}",
    # )

    add_pred = folium.raster_layers.ImageOverlay(
            image=target,
            bounds=[[polygon['latitude'][0], polygon['longitude'][0]],
                    [polygon['latitude'][2], polygon['longitude'][2]]],
            opacity=0.8,
            colormap=semantic_cmap())

    # st.session_state["markers"].append(marker)
    st.session_state["markers"].append(add_pred)
    st.session_state["center"]=[float(lon) , float(lat)]
    st.session_state["zoom"]=15

# with st.echo(code_location="below"):

##### Init Map
m = folium.Map(
    location=st.session_state["center"], zoom_start=st.session_state["zoom"]
)
m.add_child(folium.LatLngPopup())
folium.TileLayer(
    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr = 'Esri',
    name = 'Esri Satellite',
    overlay = False,
    control = True
    ).add_to(m)
#####

fg = folium.FeatureGroup(name="Markers")
for marker in st.session_state["markers"]:
    fg.add_child(marker)
m.add_child(fg)
st_folium(
    m,
    key="new",
    feature_group_to_add=fg,
    width=700,
)
