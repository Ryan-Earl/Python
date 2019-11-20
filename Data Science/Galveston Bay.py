#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# load data, format for comparison between data and metadata
meta = pd.read_csv("E:\\Galveston_Bay\\gb_metadata.csv")
meta['site_description'] = meta['Site Description ']
meta = meta[['site_description', 'Lat', 'Lon']]
meta['site_description'] = [i.replace('@', 'at').lower() for i in meta.site_description]
meta['match'] = [i.replace(' ', '') for i in meta.site_description]

data = pd.read_csv("E:\\Galveston_Bay\\bay_data.csv")
data['site_description'] = [i.lower() for i in data['Site Description']]
data = data.drop(['Site Description'], axis=1)
data['match'] = [i.replace(' ', '') for i in data.site_description]

# see what stations are in the data and not the metadata
data_set = set(data['match'])
meta_set = set(meta['match'])

data_notin_meta = data_set - meta_set
meta_notin_data = meta_set - data_set

count = 0
for i in data['match']:
    if i in data_notin_meta:
        count += 1

data = data[~data['match'].isin(data_notin_meta)]
meta = meta[~meta['match'].isin(meta_notin_data)]

# merge data and metadata
df = pd.merge(data, meta, on='match')

df.to_csv("E:\\Galveston_Bay\\cleaned_data.csv")

df = pd.read_csv("E:\\Galveston_Bay\\galv_bay.csv")
df = df.groupby('site_description').bfill()
df = df.groupby('site_description').ffill()

df.to_csv('E:\\Galveston_Bay\\station_data.csv')


# In[ ]:


df = pd.read_csv("C:\\Users\\Eileen\\Documents\\Galveston\\station_data.csv", parse_dates = ['Sample_Date'])
df = df.drop(['Unnamed: 0'], axis=1)
avg_station_sal = df.filter(['site_description', 'Lat', 'Lon', 'Salinity (ppt)']).groupby(['site_description']).mean().rename(columns={'Salinity (ppt)':'salinity_avg'})
#df.describe()
post_harvey = df[(df.Sample_Date >= '2017-09-03')]
pre_harvey = df[(df.Sample_Date < '2017-08-17')]
post_harvey_sal = post_harvey.filter(['site_description', 'Salinity (ppt)'], axis=1).groupby(['site_description']).mean().rename(columns={'Salinity (ppt)':'salinity_post_avg'})
pre_harvey_sal = pre_harvey.filter(['site_description', 'Salinity (ppt)'], axis=1).groupby(['site_description']).mean().rename(columns={'Salinity (ppt)':'salinity_pre_avg'})
station_salinity = pd.merge(pd.merge(avg_station_sal, post_harvey_sal, on='site_description'), pre_harvey_sal, on='site_description')


# In[ ]:


station_salinity.to_csv("C:\\Users\\Eileen\\Documents\\Galveston\\station_salinity.csv")


# In[38]:


import pandas as pd
from math import sqrt
from statistics import mean
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# import dataset and rename variables
df = pd.read_csv(r"C:\Users\Eileen\Documents\Galveston\station_data.csv")
df = df.rename(columns={'Water_Temp (in C)':'water temp',
                        'Avg_DO (mg/L)':'dissolved oxygen',
                        'Sample_pH':'ph',
                        'Transparency (meters)':'transparency',
                        'Salinity (ppt)':'salinity'
                       })

# dissolved oxygen
df['dissolved oxygen'] = np.where(df['dissolved oxygen'].between(0,.5), 0, df['dissolved oxygen'])
df['dissolved oxygen'] = np.where(df['dissolved oxygen'].between(.5,3), .25, df['dissolved oxygen'])
df['dissolved oxygen'] = np.where(df['dissolved oxygen'].between(3,5), .50, df['dissolved oxygen'])
df['dissolved oxygen'] = np.where(df['dissolved oxygen'].between(5,10), 1, df['dissolved oxygen'])
df['dissolved oxygen'] = df['dissolved oxygen'].mask(df['dissolved oxygen'] > 10, .75)
df = df.drop(['Unnamed: 0'], axis=1)
avg_do = df.filter(['site_description','dissolved oxygen']).groupby(['site_description']).mean()

# PH
df['ph'] = np.where(df['ph'].between(0,5), 0, df['ph'])
df['ph'] = np.where(df['ph'].between(5,6.5), .5, df['ph'])
df['ph'] = np.where(df['ph'].between(6.5,8), 1, df['ph'])
df['ph'] = np.where(df['ph'].between(8,9), .5, df['ph'])
df['ph'] = df['ph'].mask(df['ph'] > 9, 0)
avg_ph = df.filter(['site_description','ph']).groupby(['site_description']).mean()

# transparency
scaler = MinMaxScaler()
df["transparency"] = scaler.fit_transform(df[["transparency"]])
avg_trans = df.filter(['site_description','transparency']).groupby(['site_description']).mean()

# salinity
sal_dict = dict(tuple(df.filter(['site_description','salinity']).groupby('site_description').salinity))

new_dict = {}
for key, value in sal_dict.items():
    val_list = []
    for i in value:
        val_list.append(sqrt(((i-mean(value))**2.0)/len(value)))
    norm_list = []
    for i in val_list:
        norm_list.append((i-min(val_list))/(max(val_list)-min(val_list)))
    new_dict[key] = mean(norm_list)
sal_final_dict = {}
for key, value in new_dict.items():
    sal_final_dict[key] = (value-min(new_dict.values()))/(max(new_dict.values())-min(new_dict.values()))

sal_scale = pd.DataFrame(sal_final_dict, index=['salinity']).T

# temperature
scaler = MinMaxScaler()
df["water temp"] = scaler.fit_transform(df[["water temp"]])
avg_temp = df.filter(['site_description','water temp']).groupby(['site_description']).std() 

lst = []
for i in avg_temp['water temp']:
     lst.append((i-min(avg_temp['water temp']))/(max(avg_temp['water temp'])-min(avg_temp['water temp'])))
avg_temp['water temp'] = lst

# combined dataframe
combined_df = avg_temp.merge(sal_scale.merge(avg_trans.merge(avg_do.merge(avg_ph, on='site_description'), on='site_description'), left_index=True, right_index=True), left_index=True, right_index=True)
column_names = [combined_df['water temp'], combined_df['salinity'], combined_df['transparency'], combined_df['dissolved oxygen'], combined_df['ph']]
combined_df['health_index'] = pd.concat(column_names, axis=1).sum(axis=1)
health_index = []
for i in combined_df['health_index']:
    health_index.append(round(i/5, 4))
combined_df['health_index'] = health_index
combined_df['health_score'] = pd.qcut(combined_df['health_index'], 5, labels=['F','D','C','B','A'])
combined_df = combined_df.merge(df.filter(['site_description','Lat','Lon']), how='left', on='site_description')
df = combined_df.drop_duplicates(keep='first')
df = df.reset_index().drop(['index'], axis=1)


# In[39]:


df.to_csv(r"E:\\Galveston_Bay\\station_health_index.csv")


# In[ ]:




