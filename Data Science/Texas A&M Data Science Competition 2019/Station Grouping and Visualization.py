#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import datetime as datetime

#visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style as style
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
style.use('seaborn-colorblind')


# In[34]:


#creates table of stations withh their latlon to plot in arcpro and each stations lifetime count total
station = pd.read_csv('E:\TAMIDS 2019\Data\stations.csv')
#create station time ridden
time_ridden = pd.read_csv('E:\TAMIDS 2019\Data\Station_time_ridden.csv')
b = time_ridden['time_ridden'].map(lambda x: pd.to_timedelta(x).seconds)
time_ridden['time_ridden'] = b/60
time_ridden = time_ridden.groupby('start_station')['time_ridden'].sum()
#read dataframes for stations and their corresponding latlon
latlon = pd.read_csv('E:\TAMIDS 2019\Data\station_location.csv')
latlon = latlon.groupby('start_station')['station_lat', 'station_lon'].first()
#merge
station = pd.merge(station, latlon, on='start_station', how='left')
station = pd.merge(station, time_ridden, on='start_station', how='left')
station_count = pd.read_csv('E:\TAMIDS 2019\Data\start_count_ungrouped.csv')
station_count = station_count.groupby('start_station')['count'].sum()
station = pd.merge(station, station_count, on='start_station', how='left')
station = station.drop([0])


# In[36]:


station.to_csv(r'E:\TAMIDS 2019\Data\count_groupedby_station_latlon_timeridden.csv')


# In[38]:


#seperate data by region
dtla = station.where(station['Region'] == 'DTLA')
dtla = dtla[np.isfinite(dtla['count'])]
venice = station.where(station['Region'] == 'Venice')
venice = venice[np.isfinite(venice['count'])]
pla = station.where(station['Region'] == 'Port of LA')
pla = pla[np.isfinite(pla['count'])]
pasadena = station.where(station['Region'] == 'Pasadena')
pasadena = pasadena[np.isfinite(pasadena['count'])]


# In[39]:


dtla.to_csv(r'E:\TAMIDS 2019\Data\count_groupedby_station_latlon_timeridden_dtla.csv')
venice.to_csv(r'E:\TAMIDS 2019\Data\count_groupedby_station_latlon_timeridden_venice.csv')
pla.to_csv(r'E:\TAMIDS 2019\Data\count_groupedby_station_latlon_timeridden_pla.csv')
pasadena.to_csv(r'E:\TAMIDS 2019\Data\count_groupedby_station_latlon_timeridden_pasadena.csv')


# In[64]:


region = pd.read_csv(r'E:\TAMIDS 2019\Data\station_counts_only.csv')


# In[71]:


region = region.groupby('Region')['count'].sum().reset_index()
region['count']
ax = sns.barplot(data=region, x='Region', y='count').set(ylabel='Ride Count', title='Ride Count by Region')


# In[72]:


#import cleaned data
series = pd.read_csv(r"E:\TAMIDS 2019\Data\timecorrected.csv", low_memory=False, parse_dates=['Time'])


# In[77]:


results = pd.read_csv(r'E:\TAMIDS 2019\Data\Performance.csv')


# In[78]:


ax = sns.barplot(data=results, x='Model', y='RMSE (hourly)').set(ylabel='RMSE (hourly)', title='Model Performances')


# In[84]:


# plot
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(15,9)
sns.barplot(data=results, x='Model', y='RMSE (hourly)', ax=ax).set(ylabel='RMSE (hourly)', title='Model Performances')


# In[ ]:




