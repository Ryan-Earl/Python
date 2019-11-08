#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import math


# In[27]:


def Northings_and_Eastings(data, Northing, Easting):
    
    df = pd.DataFrame(data)

    delta_N = []
    delta_E = []

    for az, dist in df.itertuples(index = False):
    
        delta_N.append(math.degrees(math.cos(az))*dist)
        delta_E.append(math.degrees(math.sin(az))*dist)
    
    df['delta_N'], df['delta_E'] = delta_N, delta_E

    Northings = []
    Eastings = []

    for del_n, del_e in zip(df.delta_N, df.delta_E):
    
        Northings.append(Northing + del_n)
        Northing = Northing + del_n
        Eastings.append(Easting + del_e)
        Easting = Easting + del_e
    
    df['Northing'], df['Easting'] = Northings, Eastings
    
    return df


# In[28]:


data = {'Az':[0, 75.0989, 226.5536, 279.0861, 326.49], 'Distance':[25.49, 17.30, 42.80, 70.31, 48.47]}
Northing = 3112350.7413
Easting = 1083066.5067

Northings_and_Eastings(data, Northing, Easting)

