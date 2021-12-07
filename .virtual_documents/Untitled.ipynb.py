from shapely.geometry import Point
from shapely.ops import transform
from tqdm import tqdm
from pyproj import Geod
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapely.wkt
import geopandas as gpd

TREE_DATA = 'data/2015_Street_Tree_Census_-_Tree_Data.csv'
TEMP_DATA = 'data/Hyperlocal_Temperature_Monitoring.csv'
BUILD_DATA = 'data/building.csv'
PARK_DATA = 'data/OpenData_ParksProperties.csv'


def get_gdf(locs):
    df = gpd.GeoDataFrame({'geometry': locs}, crs='EPSG:4326')
    df = df.to_crs('EPSG:32662')
    return df

def flip(x, y): return y, x

def get_tree_data():
    print('\t - Trees')
    trees = pd.read_csv(TREE_DATA)
    trees = trees[trees.status == 'Alive']
    tree_locs = [Point(row['latitude'],row['longitude']) for _, row in trees.iterrows()]
    return get_gdf(tree_locs)

def get_building_data():
	print('\t - Buildings')
	geod = Geod(ellps="WGS84")
	build = pd.read_csv(BUILD_DATA)
	build['area'] = build.the_geom.apply(lambda x: abs(geod.geometry_area_perimeter(shapely.wkt.loads(x))[0]))
	geometry = build.the_geom.apply(lambda x: shapely.wkt.loads(x).centroid)
	build_coords = gpd.GeoDataFrame({'geometry': geometry.apply(lambda row: Point(row.y,row.x)),'bin': build['BIN'], 'height': build['HEIGHTROOF'],'area':build['area']}, crs='EPSG:4326')
	return build_coords.to_crs('EPSG:32662')

def get_park_data():
    print('\t - Parks')
    parks = pd.read_csv(PARK_DATA)
    geometry = parks.the_geom.apply(lambda x: shapely.wkt.loads(x))
    geometry = geometry.apply(lambda x: transform(flip, x))
    return  get_gdf(geometry)

def get_temp_data():
    print('\t - Temperatures')
    temp = pd.read_csv(TEMP_DATA)
    cols = ['Latitude','Longitude','Sensor.ID']
    unique_locations = pd.DataFrame([k for k,_ in temp.groupby(cols).indices.items()],columns=cols)
    print(unique_locations.shape)
    loc_coords = get_gdf([Point(row['Latitude'],row['Longitude']) for _,row in unique_locations.iterrows()])
    return temp, loc_coords, unique_locations['Sensor.ID']


def get_variables(tree_coords,build_coords,parks_coords,loc_coords,sensor_ids):
    print('~ Getting Variables ~')

    num_trees15 = []
    min_distance_park = []
    num_buildings50 = defaultdict(set)


    for i,_ in tqdm(loc_coords.iterrows(),total=loc_coords.shape[0]):
        out = tree_coords.distance(loc_coords.iloc[[i]*tree_coords.shape[0]].reset_index(0).drop(columns='index'))
        num_trees15.append(sum(out<=15))

        out = parks_coords.geometry.distance(loc_coords.iloc[[i]*parks_coords.shape[0]].reset_index(0).drop(columns='index'),align=False)
        min_distance_park.append(min(out))

        out = build_coords.geometry.distance(loc_coords.iloc[[i]*build_coords.shape[0]].reset_index(0).drop(columns='index'))
        for _,row in build_coords[out<=50].iterrows():
            num_buildings50[i].add((row['height'],row['area']))

    n = loc_coords.shape[0]
    num_build, mean_fa_ratio = [0] * n,[0] * n
    for i,v in num_buildings50.items():
        num_build[i] = (len(v))
        mean_fa_ratio[i] = np.mean([(height*3.2808)/fa for height, fa in v])
    
    loc_coords['num_trees15'] = num_trees15
    loc_coords['Sensor.ID'] = sensor_ids       
    return loc_coords



	print('~ Gathering Datasets ~')
	# tree_coords = get_tree_data()
	# build_coords = get_building_data()
	# parks_coords = get_park_data()
	#temp2, loc_coords2, sensor_ids2 = get_temp_data()
    locs = get_variables(tree_coords,build_coords,parks_coords,loc_coords,sensor_ids)
    # locs = locs.drop(columns='geometry')
    # temp = temp.merge(locs)
    # temp.to_csv('data.csv',index=False)


temps = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temps.head()

covariates = pd.read_csv('data/temp3.csv').round(8)
covariates2 = pd.read_csv('data/temp_1.csv').round(8)
covariates = covariates.merge(covariates2)
covariates['mean_fa_ratio'] = covariates['mean_fa_ratio'].fillna(0)
temps = temps.drop(index=np.where(temps['AirTemp'].isna())[0]).reset_index(0)
covariates = covariates.rename(columns={'num_trees_15m':'num_trees15'})
data = temps.merge(covariates, how='outer', on=['Latitude','Longitude'])


data.to_csv('data.csv',index=False)


data = data[pd.to_datetime(data.Day).dt.month.isin([7,8])].reset_index(0).drop(columns='index')


avg = data.groupby(['Latitude','Longitude','Hour']).mean().reset_index(0).reset_index(0).reset_index(0)








cols = ['num_trees15','mean_fa_ratio','min_distance_park','num_build500']
titles = ['Number of Trees within 15m','Mean Floor-Area Ratio','Minimum Distance to Park (m)','Number of Buildings within 50m']
fig, axes = plt.subplots(2,2,sharex=True,figsize=(10,10))
fig.suptitle('Hourly Temperatures Averaged by Hour',fontsize=16)
for i in range(2):
    for j in range(2):
        sns.lineplot(data=avg,ax=axes[i,j],x='Hour',y='AirTemp',hue=cols[j + 2*i],palette='flare').set_title(titles[j + 2*i])


plt.show()














avg_melt = pd.melt(avg,id_vars=['AirTemp','Hour'], value_vars=['num_build500','mean_fa_ratio','min_distance_park','num_trees15'])
avg_melt.head()


import folium


data = pd.read_csv('data.csv')


cols = ['Latitude','Longitude']
unique_locations = pd.DataFrame([k for k,_ in data.groupby(cols).indices.items()],columns=cols)


get_ipython().getoutput("pip install selenium")


from selenium import webdriver
import os

def save_map(m,fn):
    delay=5
    tmpurl='file://{path}/{mapfile}'.format(path=os.getcwd(),mapfile=fn + '.html')
    m.save(fn +'.html')

    browser = webdriver.Firefox()
    browser.get(tmpurl)
    #Give the map tiles some time to load
    time.sleep(delay)
    browser.save_screenshot('map' + '.png')
    browser.quit()


Hour = 10

NYC_COORD = [40.78, -73.90]
map_nyc = folium.Map(location=NYC_COORD, zoom_start=10, 
tiles='cartodbpositron', width=400, height=400)
for _,row in unique_locations.iterrows():
    folium.CircleMarker(location=(row['Latitude'], row['Longitude']),color='orange',radius=1,opacity=0.4).add_to(map_nyc)
save_map(map_nyc,'map_1')


NYC_COORD = [40.8, -73.95]
map_nyc = folium.Map(location=NYC_COORD, zoom_start=13, 
tiles='cartodbpositron', width=400, height=400)
for _,row in unique_locations.iterrows():
    folium.CircleMarker(location=(row['Latitude'], row['Longitude']),color='organge',radius=1).add_to(map_nyc)
map_nyc






