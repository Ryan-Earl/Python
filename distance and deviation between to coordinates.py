#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
from math import degrees, acos, sin, cos, atan2, fsum, radians, sqrt

auto = pd.read_csv("Documents\\auto_waypoints.csv")
waas = pd.read_csv("Documents\\waas_waypoints.csv")

def dist(df):
    
    distance = []
    for lat_1, lon_1, lat_2, lon_2 in df.itertuples(index = False):
        
        r = 6373.0
        lat1, lon1, lat2, lon2 = radians(lat_1), radians(lon_1), radians(lat_2), radians(lon_2)
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        a = sin(delta_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2)**2

        distance.append((r * (2 * atan2(sqrt(a), sqrt(1 - a))))*1000)
    return distance

auto['dist_auto'], waas['dist_waas'] = dist(auto), dist(waas)

def var_std(df, col):
    
    avg = sum(col) / len(col)
    V = [(xi - avg) for xi in col]
    V_2 = [i**2 for i in V]
    std_dev = sqrt(sum(V_2)) / (len(V_2)-1)
    
    return V, V_2, std_dev

auto['V'], auto['V_2'], auto_dev = var_std(auto, auto['dist_auto'])
waas['V'], waas['V_2'], waas_dev = var_std(waas, waas['dist_waas'])

print(auto_dev, waas_dev)

auto.to_csv("Documents\\auto_waypoints_dev.csv")
waas.to_csv("Documents\\waas_waypoints_dev.csv")


# In[ ]:




