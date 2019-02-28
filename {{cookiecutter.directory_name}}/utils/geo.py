from pyproj import Proj, transform

def gps_to_mercator(lat, long):
    return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), lat, long)
