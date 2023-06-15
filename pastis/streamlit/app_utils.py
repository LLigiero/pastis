from matplotlib import cm, colors
import requests
import ee

from pastis.params import EARTHENGINE_TOKEN, EARTHENGINE_MAIL, DEFAULT_COORDINATES

def app_init():
    #st.set_page_config(layout="wide")
    credentials = ee.ServiceAccountCredentials(EARTHENGINE_MAIL, EARTHENGINE_TOKEN)
    ee.Initialize(credentials)


def get_coordinates(address:str)->tuple:
    ''' Give an adress or coordinate
        Return the coordinate unfer format (lat,lon)'''
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params).json()

    return (response[0]['lat'], response[0]['lon'])


def get_square_dict(coordinates:tuple)->dict:
    ''' Need execute the app_init() before use it, to authenticate in the API

        Function to request the Google Earth Engine (with tuple of
        coordinates (lat,lon))

        GEE give 5 coordinates which represents the square of the zone which
        will be determine square

        Return a dictionnary:
           { 'latitude': [list of latitude],
             'longitude': [list of longitude] }'''


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


def semantic_cmap()-> object:
    ''' Function which create the same color legend of Pastis project
        Return an object which contains the 20 colors needed.'''
    c_m = cm.get_cmap('tab20')
    def_colors = c_m.colors
    cus_colors = ['#00000010'] + [def_colors[i] for i in range(1,20)] + ['w']
    semantic_cmap = colors.ListedColormap(colors = cus_colors, name='agri',N=21)

    return semantic_cmap
