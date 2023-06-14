import os
import numpy as np

import streamlit as st
import pandas as pd

from folium import raster_layers, CircleMarker
from streamlit_folium import st_folium

from pastis.params import TEST_SAT_IMG, ZOOM_MAP
from pastis.streamlit.app_utils import *


app_init()

data_15=pd.DataFrame({'latitude':[48.91405555630452,
                            48.91383839293139,
                            48.90232747239713,
                            48.90254454829243,
                            48.91405555630452
                            ],
                   'longitude':[-1.5749111956249313,
                                -1.5574466694862161,
                                -1.5577780515880353,
                                -1.575238569119885,
                                -1.5749111956249313
                                ]})

polygon=data_15

#m= print_map(polygon)
# semantic_color= semantic_cmap()

c_m = cm.get_cmap('tab20')
def_colors = c_m.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1,20)] + ['w']
semantic_color = colors.ListedColormap(colors = cus_colors, name='agri',N=21)


m = Map(location=[polygon['latitude'][0],polygon['longitude'][0]], zoom_start=ZOOM_MAP)
TileLayer(
    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr = 'Esri',
    name = 'Esri Satellite',
    overlay = False,
    control = True
    ).add_to(m)

raster_layers.ImageOverlay(
        image=np.load(os.path.join(TEST_SAT_IMG,'TARGET_10015.npy')),
        bounds=[[polygon['latitude'][0], polygon['longitude'][0]],
                [polygon['latitude'][2], polygon['longitude'][2]]],
        colormap=lambda x: (1,0,0,x)).add_to(m)


for i in range(len(polygon)):
    CircleMarker(location=[polygon['latitude'][i], polygon['longitude'][i]],
                    radius=2,
                    weight=5).add_to(m)

st_data = st_folium(m,width=725)






user_pickup_sat_temp = st.text_input('Pickup satellite template','Tour eiffel')


if st.button('click me for prediction'):
    print ('You click ! ')

    lat,lon= get_coordinates(user_pickup_sat_temp)
    coordinates=(float(lat),float(lon))
    # polygon=get_square_dict(coordinates)

    # url = 'https://taxifare.lewagon.ai/predict'
    # request=f'{url}?pickup_datetime={concat_date_time}&pickup_longitude={user_pickup_longitude}&pickup_latitude={user_pickup_latitude}&dropoff_longitude={user_dropoff_longitude}&dropoff_latitude={user_dropoff_latitude}&passenger_count={user_passenger_count}'
    # prediction = requests.get(request).json()
