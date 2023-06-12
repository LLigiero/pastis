import streamlit as st
import requests
from pandas import DataFrame
from folium import Map, Marker
from streamlit_folium import st_folium
import geemap

def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params).json()
    return (response[0]['lat'], response[0]['lon'])


default_coordinates=[48.8649224, 2.3800903]
########## st_folium ###########
# center on coordinates = [48.8649224, 2.3800903] # 16 Villa Gaudelet, Paris
m = Map(location=[48.8649224, 2.3800903], zoom_start=16,tiles='cartodbdark_matter')
Marker([48.8649224, 2.3800903], popup="Le Wagon", tooltip="Le Wagon").add_to(m)
# call to render Folium map in Streamlit
st_data = st_folium(m, width=725)
################################

# ############ st.map #############
# sat_df = DataFrame({
# 'lat': [default_coordinates[0]],
# 'lon': [default_coordinates[1]]
# })
# st.map(sat_df)
# #################################

user_pickup_sat_temp = st.text_input('Pickup satellite template','16 Villa Gaudelet, Paris')

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









#predict?pickup_datetime=2014-07-06%2019:18:00&pickup_longitude=-73.950655&
# pickup_latitude=40.783282&dropoff_longitude=-73.984365&
# dropoff_latitude=40.769802&passenger_count=2

# pickup_datetime	DateTime	2013-07-06 17:18:00
# pickup_longitude	float	-73.950655
# pickup_latitude	float	40.783282
# dropoff_longitude	float	-73.950655
# dropoff_latitude	float	40.783282
# passenger_count	int	2
