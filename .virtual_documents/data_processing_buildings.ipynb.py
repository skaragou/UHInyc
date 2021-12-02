import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from tqdm import tqdm
import shapely.wkt
from pyproj import Geod
from collections import defaultdict

from shapely.geometry import Point,MultiPolygon


def get_gdf(locs):
    df = gpd.GeoDataFrame({'geometry': locs}, crs='EPSG:4326')
    df = df.to_crs('EPSG:32662')
    return df


geod = Geod(ellps="WGS84")


build = pd.read_csv('data/building.csv')
build.head()


build['area'] = build.the_geom.apply(lambda x: abs(geod.geometry_area_perimeter(shapely.wkt.loads(x))[0]))


temp = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temp.head()


unique_locations = temp.groupby(['Latitude','Longitude']).count().reset_index(0).reset_index(0)[['Latitude','Longitude']]
temp_locs = [Point(row['Latitude'],row['Longitude']) for _,row in unique_locations.iterrows()]

temp_coord = get_gdf(temp_locs)


geometry = build.the_geom.apply(lambda x: shapely.wkt.loads(x).centroid)
build_coords = gpd.GeoDataFrame({'geometry': geometry.apply(lambda row: Point(row.y,row.x)),'bin': build['BIN'], 'height': build['HEIGHTROOF'],'area':build['area']}, crs='EPSG:4326')
build_coords = build_coords.to_crs('EPSG:32662')


num_buildings50 = defaultdict(set)
for i,_ in tqdm(temp_coord.iterrows(),total=temp_coord.shape[0]):
    out = build_coords.geometry.distance(temp_coord.iloc[[i]*build.shape[0]].reset_index(0).drop(columns='index'))
    for _,row in build_coords[out<=50].iterrows():
        num_buildings50[i].add((row['height'],row['area']))


n = temp_coord.shape[0]
num_build, mean_fa_ratio = [0] * n,[0] * n
for i,v in num_buildings50.items():
    num_build[i] = (len(v))
    mean_fa_ratio[i] = np.mean([(height*3.2808)/fa for height, fa in v])        


unique_locations['num_build50'] = num_build
unique_locations['mean_fa_ratio'] = mean_fa_ratio


unique_locations.to_csv('temp2.csv',index=False)


unique_locations



