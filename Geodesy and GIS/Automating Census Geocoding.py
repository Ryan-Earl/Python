#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# for outside users
# data_path = input(r"Enter data path: ") 
# df = pd.read_csv(data_path)

# clean input data, create unique index and reorder columns
df = pd.read_csv(r"C:\Users\Eileen\Documents\T3 project\t3_test_data.csv")
df['Number'] = range(1, len(df) +1)
df = df.rename(columns={'House Number - Street Name': 'Street Address'})
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
id_store = df['ID']
df_export = df.drop(['ID'], axis=1)
df_export.to_csv(r"C:\Users\Eileen\Documents\T3 project\geocoder_data.csv", index=False)


# In[ ]:


import pandas as pd
import numpy as np
import warnings

# ignore deprecation warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

geocode_results = pd.read_csv(r"C:\Users\Eileen\Documents\T3 project\census_geocode_results.csv", na_values=["No_Match"])
no_match_count = (geocode_results.is_match.isnull().sum())
match_percent = ((len(geocode_results.is_match)) - no_match_count)/(len(geocode_results.is_match))

geocode_matches = geocode_results.dropna()
cords = [x.split(',') for x in geocode_matches.lat_lon]
lat, lon = [], []
for point in cords:
    lat.append(point[0])
    lon.append(point[1])
    
geocode_matches['lat'], geocode_matches['lon'] = lat, lon
geocode_matches = geocode_matches.drop(['lat_lon'], axis = 1)
geocode_matches.to_csv(r"C:\Users\Eileen\Documents\T3 project\census_geocode_matches.csv", index=False)


# In[ ]:


#-------------------Pre-Processing------------------

import pandas as pd
import numpy as np

df = pd.read_csv(r"E:\\T3 project\\T3_disclosures\\sheet_block_addresses_m3360TX.csv", dtype={'ed2sb':str})

df['ed2sb'] = df['ed2sb'].str[1:]
df['ed2a'] = [x[0:4] for x in df.ed2sb]
df['ed2b'] = [x[4:7] for x in df.ed2sb]
df['Index'] = range(1, len(df) +1)
houston = ['Houston'] * len(df)
state = ['Texas'] * len(df)
df['City'] = houston
df['State'] = state
df['Zip Code'] = ''
df = df.rename(columns={'street2': 'Street Address'})
df.to_csv("E:\\T3 project\\pre_file\\pre_file.csv")
df2 = df.filter(['Index', 'Street Address', 'City', 'State', 'Zip Code'])

# split df2 into batches of 10,000
size = 9999
list_of_dfs = [df2.loc[i:i+size-1,:] for i in range(0, len(df2),size)]

# save batches to files
num = '1'
for df in list_of_dfs:
    df.to_csv(f"E:\\T3 project\\full_houston_df_batches\\df_batch_{num}.csv", index=False)
    num = str(int(num)+1)


# In[1]:


#################################### Functioning base geocode processing ################################
import pandas as pd
import numpy as np
import re

import os
from os import listdir
from os.path import isfile, join

def sorted_alphanumeric(data):
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    
    return sorted(data, key=alphanum_key)

def create_dataframe(mypath, prepath):
    
    pre_file = pd.read_csv(prepath).filter(['ed2b', 'Index'], axis=1).rename(columns={'Index':'index'})
    
    only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    sorted_files = sorted_alphanumeric(only_files)
    df_list = [pd.read_csv(mypath+'\\'+sorted_files[i]) for i in range(len(sorted_files))]
    
    for i in range(len(df_list)):
        df_list[i].columns = df_list[0].columns
        
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.merge(pre_file, on='index')
    df['ed2b'] = df['ed2b'].astype(str)
    ed = df['ed2b'].iloc[0]
    
    ed_count, success_count = 0.0, 0.0
    ed_dict = {}
    for index, row in df.iterrows():
        
        if ed == row['ed2b']:
            ed_count += 1
            if row['success'] == 'Match':
                success_count += 1
        else:
            ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
            ed_count, success_count = 1.0, 0.0
            ed = row['ed2b']
            if row['success'] == 'Match':
                success_count += 1
                
    ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
    ed_matches = pd.DataFrame(list(ed_dict.items()), columns=['ed2b', 'match_percent'])
    ed_matches['match_percent'] = (ed_matches['match_percent'] > 50.0) * 1
    
    df = df.merge(ed_matches, on='ed2b')
    
    return df

def get_matches(df):
    
    df = df.dropna()
    cords = [x.split(',') for x in df.latlon]
    
    lat, lon = [], []
    for point in cords:
        lat.append(point[0])
        lon.append(point[1])
    
    df['lat'], df['lon'] = lat, lon
    df = df.drop(['latlon'], axis = 1).filter(['index', 'geocoded address', 'side of street', 'ed2b', 'lat', 'lon', 'match_percent'], axis=1)
    
    return df
    
def main():
    
    df = create_dataframe('E:\\T3 project\\full_houston_geocode_results', 'E:\\T3 project\\pre_file\\pre_file.csv')
    only_matches = get_matches(df)
    
    return only_matches

test = main()


# In[ ]:


test.to_csv("E:\\T3 project\\full_houston_df_batches\\test2.csv", index=False)


# In[ ]:


test.head()


# In[ ]:


############################ Import to run in ArcPro, To Finish ##################################33

import arcpy
import pandas as pd
import numpy as np
import re

import os
from os import listdir
from os.path import isfile, join


class Toolbox(object):
    def __init__(self):
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [DisplayGeocode]

class DisplayGeocode(object):
    def __init__(self):
        self.label = "Tool"
        self.description = "converts folder of batch census geocode results to a single cleaned dataframe, then displays points by lat/lon"
        self.canRunInBackground = False

    def getParameterInfo(self):
        
        batch_path = arcpy.Parameter(
            displayName = "File Path to the folder containing Geocode Results",
            name = "BatchPath",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input"
        )
        pre_path = arcpy.Parameter(
            displayName = "File path to your csv pre geocoding",
            name = "PrePath",
            datatype = "DeFile",
            parameterType = "Required",
            direction = "Input",
        )
        post_path = arcpy.Parameter(
            displayName = "File path for you new merged csv file",
            name = "PrePath",
            datatype = "DeFile",
            parameterType = "Required",
            direction = "Input",
        )
        gdb_path = arcpy.Parameter(
            displayName = "GDB Folder Path",
            name = "GDBFolder",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input"
        )
        gdb_name = arcpy.Parameter(
            displayName = 'GDB Name',
            name = 'GDBName',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        params = [batch_path, pre_path, post_path, gdb_path, gdb_name]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def sorted_alphanumeric(self, data):
    
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        
        return sorted(data, key=alphanum_key)

    def create_dataframe(self, prepath, mypath):
    
        pre_file = pd.read_csv(prepath).filter(['ed2b', 'Index'], axis=1).rename(columns={'Index':'index'})
        only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        sorted_files = self.sorted_alphanumeric(only_files)
        df_list = [pd.read_csv(mypath+'\\'+sorted_files[i]) for i in range(len(sorted_files))]
    
        for i in range(len(df_list)):
            df_list[i].columns = df_list[0].columns
        
        df = pd.concat(df_list).reset_index(drop=True)
        df = df.merge(pre_file, on='index')
        df['ed2b'] = df['ed2b'].astype(str)
        ed = df['ed2b'].iloc[0]
        ed_count, success_count = 0.0, 0.0
        ed_dict = {}
        
        for index, row in df.iterrows():
            if ed == row['ed2b']:
                ed_count += 1
                if row['success'] == 'Match':
                    success_count += 1
                else:
                    ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
                    ed_count, success_count = 1.0, 0.0
                    ed = row['ed2b']
                    if row['success'] == 'Match':
                        success_count += 1
                
        ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
        ed_matches = pd.DataFrame(list(ed_dict.items()), columns=['ed2b', 'match_percent'])
        ed_matches['match_percent'] = (ed_matches['match_percent'] > 50.0) * 1
        df = df.merge(ed_matches, on='ed2b')
    
        return df

    def get_matches(self, df):
    
        df = df.dropna()
        cords = [x.split(',') for x in df.latlon]
    
        lat, lon = [], []
        for point in cords:
            lat.append(point[0])
            lon.append(point[1])
    
        df['lat'], df['lon'] = lat, lon
        df = df.drop(['latlon'], axis = 1).filter(['index', 'geocoded address', 'side of street', 'ed2b', 'lat', 'lon', 'match_percent'], axis=1)
    
        return df
    
    def save_df(self, parameters)
    
        df = self.create_dataframe(f"{parameters[0].valueAsText}", f"{parameters[1].valueAsText}")
        only_matches = self.get_matches(df)
        only_matches.to_csv(f"{parameters[2].valueAsText}"), index=False)
        
    def execute(self, parameters, messages):
        
        self.save_df(parameters)
        return


# In[5]:


############### Match accuracy exact by Enumeration District #####################3

import pandas as pd
import numpy as np
import re

import os
from os import listdir
from os.path import isfile, join

def sorted_alphanumeric(data):
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    
    return sorted(data, key=alphanum_key)

def create_dataframe(mypath, prepath):
    
    pre_file = pd.read_csv(prepath).filter(['ed2b', 'Index'], axis=1).rename(columns={'Index':'index'})
    
    only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    sorted_files = sorted_alphanumeric(only_files)
    df_list = [pd.read_csv(mypath+'\\'+sorted_files[i]) for i in range(len(sorted_files))]
    
    for i in range(len(df_list)):
        df_list[i].columns = df_list[0].columns
        
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.merge(pre_file, on='index')
    df['ed2b'] = df['ed2b'].astype(str)
    ed = df['ed2b'].iloc[0]
    
    ed_count, success_count = 0.0, 0.0
    ed_dict = {}
    for index, row in df.iterrows():
        
        if ed == row['ed2b']:
            ed_count += 1
            if row['success'] == 'Match':
                success_count += 1
        else:
            ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
            ed_count, success_count = 1.0, 0.0
            ed = row['ed2b']
            if row['success'] == 'Match':
                success_count += 1
                
    ed_dict[f'{ed}'] = round((success_count/ed_count)*100, 2)
    ed_matches = pd.DataFrame(list(ed_dict.items()), columns=['ed2b', 'match_percent'])
    
    return ed_matches

df = create_dataframe('E:\\T3 project\\full_houston_geocode_results', 'E:\\T3 project\\pre_file\\pre_file.csv')


# In[20]:


################ ED accuracy ######################

notable_ed = ['284', '213', '286', '206', '210', '261', '287', '288', '289']
print((df.loc[df['ed2b'].isin(notable_ed)]).rename(columns={'ed2b':'ED', 'match_percent':'Match Accuracy'}).to_string(index=False))
print("---------------------------------")
print("Notable Enumeration District Mean Accuracy")
print(round(df.match_percent.mean(), 2))

