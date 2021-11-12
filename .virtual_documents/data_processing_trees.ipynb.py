import pandas as pd
import numpy as np
import folium
import geopandas as gpd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Point
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


trees = pd.read_csv('data/2015_Street_Tree_Census_-_Tree_Data.csv')


trees.head()


trees.shape


temp = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temp.head()


temp.shape


unique_locations = temp.groupby(['Latitude','Longitude']).count().reset_index(0).reset_index(0)[['Latitude','Longitude']]


def get_gdf(locs):
    df = gpd.GeoDataFrame({'geometry': locs}, crs='EPSG:4326')
    df = df.to_crs('EPSG:2263')
    return df


tree_locs = [Point(row['latitude'],row['longitude']) for _,row in trees.iterrows()]
temp_locs = [Point(row['Latitude'],row['Longitude']) for _,row in unique_locations.iterrows()]

tree_coord = get_gdf(tree_locs)
temp_coord = get_gdf(temp_locs)


num_trees100 = []
for i,_ in tqdm(temp_coord.iterrows(),total=temp_coord.shape[0]):
    out = tree_coord.distance(temp_coord.iloc[[i]*trees.shape[0]].reset_index(0).drop(columns='index')) / 3.2808
    num_trees100.append(sum(out<=100))


unique_locations['num_trees_50m'] = num_trees100


unique_locations.to_csv('data/temp_1.csv',index=False)


unique_locations


temp_2 = pd.read_csv('data/temp2.csv')


temp_2['mean_fa_ratio'] = temp_2['mean_fa_ratio'].fillna(0)


unique_locations['num_build500'] = temp_2['num_build500']
unique_locations['mean_fa_ratio'] = temp_2['mean_fa_ratio']


days = temp.groupby(['Latitude','Longitude','Hour']).mean('AirTemp').reset_index(0).reset_index(0).reset_index(0)


plt.figure(figsize=(10,10))
for i,row in unique_locations.iterrows():
    out = days[(days['Latitude'] == row["Latitude"]) & (days['Longitude'] == row["Longitude"])]
    sns.lineplot(data=out,x='Hour',y='AirTemp')


plt.figure(figsize=(10,10))

x,y,h = [],[],[]

for i,row in unique_locations.iterrows():
    out = days[(days['Latitude'] == row["Latitude"]) & (days['Longitude'] == row["Longitude"])]
    x.append(out['Hour'].values)
    y.append(out['AirTemp'].values)
    h.append([int(row['num_trees_50m'])] * out.shape[0])
    
ax = sns.lineplot(x=np.concatenate(x),y=np.concatenate(y),hue=np.concatenate(h))
ax.set_title('Number of Trees within 50m')


plt.figure(figsize=(10,10))

x,y,h = [],[],[]

for i,row in unique_locations.iterrows():
    out = days[(days['Latitude'] == row["Latitude"]) & (days['Longitude'] == row["Longitude"])]
    x.append(out['Hour'].values)
    y.append(out['AirTemp'].values)
    h.append([int(row['num_build500'])] * out.shape[0])
    
ax = sns.lineplot(x=np.concatenate(x),y=np.concatenate(y),hue=np.concatenate(h))
ax.set_title('Number of buildings with 500m')


plt.figure(figsize=(10,10))

x,y,h = [],[],[]

for i,row in unique_locations.iterrows():
    out = days[(days['Latitude'] == row["Latitude"]) & (days['Longitude'] == row["Longitude"])]
    x.append(out['Hour'].values)
    y.append(out['AirTemp'].values)
    h.append([row['mean_fa_ratio']] * out.shape[0])
    
ax = sns.lineplot(x=np.concatenate(x),y=np.concatenate(y),hue=pd.cut(np.concatenate(h),10))
ax.set_title('Mean Floor-Area ratio with 500m')


cuts = pd.cut(list(range(int(temp['AirTemp'].min()),int(temp['AirTemp'].max()) + 1)),5)


cuts.categories


colors = ['#00f4ff','#00ff6a','#F8ff00','#Ffa600','#Ff0000']


airtemp = temp.groupby(['Latitude','Longitude','Hour']).agg({'AirTemp':np.max})


airtemp = airtemp.reset_index(0).reset_index(0).reset_index(0)


airtemp


(int(temp['AirTemp'].max()) + 1 - int(temp['AirTemp'].min()))/5


Hour = 10

NYC_COORD = [40.7128, -74.0059]
map_nyc = folium.Map(location=NYC_COORD, zoom_start=12, 
tiles='cartodbpositron', width=800, height=800)
for _,row in airtemp[airtemp['Hour'] == Hour].iterrows():
    html = str(row['AirTemp'])

    iframe = folium.IFrame(html,
                           width=100,
                           height=100)

    popup = folium.Popup(iframe,
                         max_width=100)

    c = 0
    for i in range(len(cuts.categories)):
        if row['AirTemp'] in cuts.categories[i]:
            c =i
    
    folium.CircleMarker(location=(row['Latitude'], row['Longitude']),color=colors[c],radius=1).add_to(map_nyc)
map_nyc


interact(get_map, Hour=widgets.IntSlider(min=0, max=24, step=1))




