from matplotlib import cm, colors
import requests
import ee

from folium import Map, TileLayer, raster_layers, CircleMarker, LatLngPopup, Marker

from pastis.params import EARTHENGINE_TOKEN, EARTHENGINE_MAIL, ZOOM_MAP

def app_init():
    #st.set_page_config(layout="wide")
    credentials = ee.ServiceAccountCredentials(EARTHENGINE_MAIL, EARTHENGINE_TOKEN)
    ee.Initialize(credentials)

def map_init():
    m = Map(location=(47.78530809282205, 2.6266160414864257), zoom_start=6)
    m.add_child(LatLngPopup())
    TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
        ).add_to(m)
    return m


def update_map(coordinate, polygon,target):
    print (f'polygon_in_update={polygon}')
    m_=map_init()
    Marker(coordinate, tooltip='Le Wagon Paris').add_to(m_)
    raster_layers.ImageOverlay(
            image=target,
            bounds=[[polygon['latitude'][0], polygon['longitude'][0]],
                    [polygon['latitude'][2], polygon['longitude'][2]]],
            opacity=0.5,
            colormap=semantic_cmap()).add_to(m_)

    for i in range(len(polygon)):
        CircleMarker(location=[polygon['latitude'][i], polygon['longitude'][i]],
                        radius=2,
                        weight=5).add_to(m_)

    return m_

def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params).json()
    return (response[0]['lat'], response[0]['lon'])

def get_square_dict(coordinates): #coordinates = Tuple de float (lat,lon)
    square_dict={'latitude':[],
        'longitude':[]}

    point = ee.Geometry.Point(coordinates) # coordinates should be list, try with tuple
    areaM2 = 1280*1280
    square = point.buffer(ee.Number(areaM2).sqrt().divide(2), 1).bounds()

    square_list=square.getInfo()['coordinates'][0]

    for i in range(len(square_list)):
        square_dict['latitude'].append(square_list[i][1])
        square_dict['longitude'].append(square_list[i][0])

    return(square_dict)

def semantic_cmap():
    c_m = cm.get_cmap('tab20')
    def_colors = c_m.colors
    cus_colors = ['k'] + [def_colors[i] for i in range(1,20)] + ['w']
    semantic_cmap = colors.ListedColormap(colors = cus_colors, name='agri',N=21)
    return semantic_cmap
