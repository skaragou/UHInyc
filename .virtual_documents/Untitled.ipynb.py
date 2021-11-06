import pandas as pd
import numpy as np
import folium


temp = pd.read_csv('data/Hyperlocal_Temperature_Monitoring.csv')
temp.head()


days = temp.groupby(['Latitude','Longitude','Day'])


days


airtemp = temp.groupby(['Latitude','Longitude']).agg({'AirTemp':np.max})


airtemp = airtemp.reset_index(0).reset_index(0) 


airtemp


NYC_COORD = [40.7128, -74.0059]
map_nyc = folium.Map(location=NYC_COORD, zoom_start=12, 
tiles='cartodbpositron', width=800, height=800)
for _,row in airtemp.iterrows():
    html = str(row['AirTemp'])

    iframe = folium.IFrame(html,
                           width=100,
                           height=100)

    popup = folium.Popup(iframe,
                         max_width=100)

    folium.CircleMarker(location=(row['Latitude'], row['Longitude']),popup=popup,radius=1).add_to(map_nyc)
map_nyc


lat_lon



