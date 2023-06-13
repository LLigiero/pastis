import streamlit as st
import requests
from pandas import DataFrame
from folium import Map, Marker
from streamlit_folium import st_folium
import geemap

from pastis.params import EARTHENGINE_TOKEN, EARTHENGINE_MAIL, TEST_SAT_IMG, ZOOM_MAP


def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params).json()
    return (response[0]['lat'], response[0]['lon'])


user_pickup_sat_temp = st.text_input('Pickup satellite template','16 Villa Gaudelet, Paris')
import ee
credentials = ee.ServiceAccountCredentials(EARTHENGINE_MAIL, EARTHENGINE_TOKEN)
ee.Initialize(credentials)









if st.button('click me for prediction'):
    print ('You click ! ')

    lat,lon= get_coordinates(user_pickup_sat_temp)
    print (lat,type(lon))
    lat=float(lat)
    lon=float(lon)
    print (lat,type(lon))

    # url = 'https://taxifare.lewagon.ai/predict'
    # request=f'{url}?pickup_datetime={concat_date_time}&pickup_longitude={user_pickup_longitude}&pickup_latitude={user_pickup_latitude}&dropoff_longitude={user_dropoff_longitude}&dropoff_latitude={user_dropoff_latitude}&passenger_count={user_passenger_count}'
    # prediction = requests.get(request).json()
