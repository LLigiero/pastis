import streamlit as st
import ee, requests, os
import pandas as pd
import geemap.foliumap as geemap

from pastis.params import EARTHENGINE_TOKEN, EARTHENGINE_MAIL, TEST_SAT_IMG, ZOOM_MAP

    # - **Web App:** <https://gishub.org/geemap-apps>
    # - **Github:** <https://github.com/giswqs/geemap-apps>

start_coordinates=(48.91405555630452,-1.5749111956249313)# S2_100015 first coordinate


def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params).json()
    return (response[0]['lat'], response[0]['lon'])



def init_page():
    data=pd.DataFrame({'latitude':[48.91405555630452,
                               48.91383839293139,
                               48.90254454829243,
                               48.91405555630452],
                   'longitude':[-1.5749111956249313,
                                -1.5574466694862161,
                                -1.575238569119885,
                                -1.5749111956249313]})

    st.set_page_config(layout="wide")
    st.title("Pastis")
    credentials = ee.ServiceAccountCredentials(EARTHENGINE_MAIL, EARTHENGINE_TOKEN)
    ee.Initialize(credentials)

    map = geemap.Map(zoom=ZOOM_MAP)
    map.add_basemap("SATELLITE")
    map.set_center(data['longitude'][0],data['latitude'][0],zoom=ZOOM_MAP)
    map.add_markers_from_xy(data, x='longitude', y='latitude')

    return map


def app(map):

    col1, _, col2, _ = st.columns([3, 0.3, 2, 1])

    with col1:
        user_pickup_sat_temp = st.text_input('Pickup satellite template','Tour eiffel')

    with col2:
        opacity = st.slider("Opacity", min_value=0.0, max_value=1.0,
                            value=0.8, step=0.05)


    landsat = os.path.join(TEST_SAT_IMG,'TARGET_10015.npy')
    map.add_raster(landsat, bands=[5, 4, 3], layer_name='Landsat')


    if st.button('click me for prediction'):
        #Map.remove_drawn_features()
        print ('You click ! ')
        lat,lon= get_coordinates(user_pickup_sat_temp)
        new_coordinate=(lat,lon)

        map.set_center(lon,lat,zoom=ZOOM_MAP)

        map.add_marker(new_coordinate, popup=user_pickup_sat_temp)
        map.addLayer(ee_object, vis_params, name, shown, opacity)
    # Render the map using streamlit

    map.to_streamlit()


##############################################################################""

map= init_page()
app(map)
