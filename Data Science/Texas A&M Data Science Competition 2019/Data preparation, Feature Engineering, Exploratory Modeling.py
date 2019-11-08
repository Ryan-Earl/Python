#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pylab
import calendar 
import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import matplotlib
from datetime import datetime
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.style as style
import seaborn as sns


# In[ ]:


#load data
bike = pd.read_csv("D:\TAMIDS 2019\GIS\LABikeData.csv", low_memory = False, parse_dates=['start_time', 'end_time'])
bikex = pd.read_csv("D:\TAMIDS 2019\GIS\LABikeData.csv", low_memory = False,)
bike['dates'] = bikex['start_time']
bike['end_dates'] = bikex['end_time']
bike1 = bike.copy(deep = True)
bike1 = bike1.drop(bike1.columns[[7,8,9,10,11,12]], axis=1)

#find missing data
print('Number of null values: '); print(bike1.isnull().sum())
print(" ")
bike1.describe(include = 'all')
print(" ")
`
#create time variables
bike1["true_date"] = bike1.dates.apply(lambda x : x.split()[0])
bike1["start_minute"] = bike1.dates.apply(lambda x : x.split()[1].split()[0] + x[-2:])
bike1["end_minute"] = bike1.end_dates.apply(lambda x : x.split()[1].split()[0] + x[-2:])
bike1['start_hour'] = bike1.start_time.dt.hour
bike1['end_hour'] = bike1.end_time.dt.hour
bike1['day'] = bike1.start_time.dt.day
bike1['month'] = bike1.start_time.dt.month
bike1['year'] = bike1.start_time.dt.year
bike1["day_name"] = bike1.true_date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%d/%m/%Y").weekday()])

# Correct the hour minute second formatting
start_minute = []
end_minute = []

# correct hours if len() is incorrect
for row in bike1['start_minute']:
    if len(row) < 10:
        start_minute.append(str(0) + row)
    else:
        start_minute.append(row)
for row in bike1['end_minute']:
    if len(row) < 10:
        end_minute.append(str(0) + row)
    else:
        end_minute.append(row)
bike1['start'] = start_minute
bike1['end'] = end_minute

# Function to convert the date format to 24 hr
start_min = []
end_min = []

for hour in bike1["start"]:  
    # Checking if last two elements of time 
    # is AM and first two elements are 12 
    if hour[-2:] == "AM" and hour[:2] == "12": 
        start_min.append(str(00) + hour[2:-2])     
    # remove the AM  
    elif hour[-2:] == "AM": 
        start_min.append(hour[:-2])   
    # Checking if last two elements of time 
    # is PM and first two elements are 12   
    elif hour[-2:] == "PM" and hour[:2] == "12": 
        start_min.append(hour[:-2]) 
    else: 
        # add 12 to hours and remove PM 
        start_min.append(str(int(hour[:2]) + 12) + hour[2:8])
for hour in bike1["end"]:  
    # Checking if last two elements of time 
    # is AM and first two elements are 12 
    if hour[-2:] == "AM" and hour[:2] == "12": 
        end_min.append(str(00) + hour[2:-2])   
    # remove the AM  
    elif hour[-2:] == "AM": 
        end_min.append(hour[:-2]) 
    # Checking if last two elements of time 
    # is PM and first two elements are 12   
    elif hour[-2:] == "PM" and hour[:2] == "12": 
        end_min.append(hour[:-2]) 
    else: 
        # add 12 to hours and remove PM 
        end_min.append(str(int(hour[:2]) + 12) + hour[2:8])

#create 24 hour time columns
bike1['end'] = end_min
bike1['start'] = start_min

#create ride time column                                        
FMT = '%H:%M:%S'
bike1['time_ridden'] = bike1['end'].apply(lambda x: datetime.strptime(x, FMT))- bike1['start'].apply(lambda x: datetime.strptime(x, FMT))

#correct multiday ride elapsed time
from datetime import timedelta
x = []
for time in bike1['time_ridden']:
    if time.days < 0:
        time = timedelta(days=0, seconds=time.seconds)
        x.append(time)
    else: 
        x.append(time)
bike1['time_ridden'] = x

#create formatted time column to merge weather data
joinkey = bike1['start_time']
new = joinkey.apply(lambda dt: dt.replace(minute=0,second=0))
bike1['joinkey'] = new

#Create season variable
#1=spring 2=summer 3=fall 4=winter
season = []
# For each row in the column,
for row in bike1['month']:
    if (row==3) or (row==4) or (row==5):
        season.append(1)
    elif (row==6) or (row==7) or (row==8):
        season.append(2)
    elif (row==9) or (row==10) or (row==11):
        season.append(3)
    else:
        season.append(4)        
#Create season
bike1['season'] = season


# In[ ]:





# In[ ]:


##Area Decomposition##


# In[ ]:





# In[41]:


#Read Data
hist_performance = pd.read_csv('E:\TAMIDS 2019\Data\historical_start.csv')
station_areas = pd.read_csv('E:\TAMIDS 2019\Data\stations.csv')
latlon = pd.read_csv('E:\TAMIDS 2019\Data\station_location.csv')
latlon = latlon.groupby('start_station')['station_lat', 'station_lon'].first()
#Convert time ridden to operable format from datetime
b = hist_performance['time_ridden'].map(lambda x: pd.to_timedelta(x).seconds)
hist_performance['time_ridden'] = b/60
#Create columns for each type of pass
walk_up = []
monthly = []
flex = []
annual = []
for row in hist_performance['passholder_type']:
    if row == "Walk-up" or row == 'One Day Pass':
        walk_up.append(1)
    else: 
        walk_up.append(0)
for row in hist_performance['passholder_type']:
    if row == 'Monthly Pass':
        monthly.append(1)
    else: 
        monthly.append(0)
for row in hist_performance['passholder_type']:
    if row == "Flex Pass":
        flex.append(1)
    else: 
        flex.append(0)
for row in hist_performance['passholder_type']:
    if row == 'Annual Pass':
        annual.append(1)
    else:
        annual.append(0) 
hist_performance['Walk_Up'] = walk_up
hist_performance['Monthly'] = monthly
hist_performance['Flex Pass'] = flex
hist_performance['Annual'] = annual
#Group by station
stations = hist_performance.groupby('start_station')['count', 'time_ridden'].sum()
#Merge station Location
stations = pd.merge(stations, station_areas['Station_ID', 'Region'], on='start_station', how='left')


# In[36]:


station_areas = station_areas.drop_duplicates()


# In[ ]:





# In[ ]:


###VISUALS###


# In[ ]:





# In[5]:


#import libraries
import pylab
import calendar 
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import statsmodels.api as sm
from datetime import datetime 
from math import sqrt
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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


# In[ ]:


#Import Data and create raw time ridden column
p = pd.read_csv("E:\TAMIDS 2019\Data\Bikes.csv", low_memory = False)
a = p['time_ridden'].map(lambda x: pd.to_timedelta(x).seconds)
p['time_ridden'] = a/60


# In[ ]:


#Cost Function based on time ridden and pass type
cost=[]
for index, row in p.iterrows():
    if row['plan_duration'] == 1 and row['passholder_type'] == "Walk-up":
        cost.append((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0)) * 1.75)
    elif row['plan_duration'] == 0 and row['passholder_type'] == "Walk-up":
        cost.append((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0)) * 3.5)
    elif row['plan_duration'] == "One Day Pass":
        if row['time_ridden'] <= 30:
            cost.append(0)
        else:
             cost.append(((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0))-1) * 1.75)
    elif row['plan_duration'] == "Monthly Pass":
        if row['time_ridden'] <= 30:
            cost.append(0)
        else:
             cost.append(((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0))-1) * 1.75)
    elif row['plan_duration'] == "Flex Pass":
        cost.append((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0)) * 1.75)
    else:
        if row['time_ridden'] <= 30:
            cost.append(0)
        else:
             cost.append(((int(row['time_ridden'] / 30) + (row['time_ridden'] % 30 > 0))-1) * 1.75)
p['total_cost'] = cost
#Export Data
p.to_csv(r'E:\TAMIDS 2019\Data\cost.csv')


# In[ ]:


#Create new Time Columns and Join data by Time
mod = pd.read_csv("E:\TAMIDS 2019\GIS\Sum.csv", low_memory = False,)


# In[ ]:


#Create columns for each type of pass
walk_up = []
monthly = []
flex = []
annual = []
for row in mod['passholder_type']:
    if row == "Walk-up" or row == 'One Day Pass':
        walk_up.append(1)
    else: 
        walk_up.append(0)
for row in mod['passholder_type']:
    if row == 'Monthly Pass':
        monthly.append(1)
    else: 
        monthly.append(0)
for row in mod['passholder_type']:
    if row == "Flex Pass":
        flex.append(1)
    else: 
        flex.append(0)
for row in mod['passholder_type']:
    if row == 'Annual Pass':
        annual.append(1)
    else:
        annual.append(0) 
mod['Walk_Up'] = walk_up
mod['Monthly'] = monthly
mod['Flex Pass'] = flex
mod['Annual'] = annual

#convert time ridden to operable format from datetime
b = mod['time_ridden'].map(lambda x: pd.to_timedelta(x).seconds)
mod['time_ridden'] = b/60

#correct Joinkey Time format
x = []
for row in mod['Joinkey']:
    x.append(row + ":00")
mod['Joinkey']=x

#group by Hour
mod = mod.groupby(['Joinkey'])[["time_ridden", "Walk_Up", "Monthly", "Flex Pass", 'Annual', 'total_cost']].sum()
#export dataframe to be cleaned
mod.to_csv(r'E:\TAMIDS 2019\Data\mod.csv')


# In[ ]:


#Create Summed time columns and join to weather data
df = pd.read_csv("E:\TAMIDS 2019\Data\mod.csv", low_memory = False, parse_dates=['Time'])


# In[ ]:


#create predictive variables
df['hour'] = df.Time.dt.hour
df['month'] = df.Time.dt.month

#Create season variable
#1=spring 2=summer 3=fall 4=winter
season = []
# For each row in the column,
for row in df['month']:
    if (row==3) or (row==4) or (row==5):
        season.append(1)
    elif (row==6) or (row==7) or (row==8):
        season.append(2)
    elif (row==9) or (row==10) or (row==11):
        season.append(3)
    else:
        season.append(4)      
#Create season
df['season'] = season

#import weather data
weather = pd.read_csv('E:\TAMIDS 2019\GIS\climate.csv', low_memory = False, parse_dates=['Time'])

#find missing data
print('Number of null values: '); print(weather.isnull().sum())
print(" ")
weather.weather_main.unique()

#merge weather and time data
result = pd.merge(df,
                 weather[['Time', 'temperature', 'humidity', 'wind_speed', 'weather_main']],
                 on='Time',
                 how='left')
#Fill missing values
result = result.fillna(method='bfill')
#export data
result.to_csv(r'E:\TAMIDS 2019\Data\result.csv')


# In[6]:


#Import Data and remove outliers
model = pd.read_csv(r"E:\TAMIDS 2019\Data\pred.csv", low_memory=False, parse_dates=["Time"])
model = model[np.abs(model["Count"]-model["Count"].mean())<=(model["Count"].std()*3)]


# In[11]:


#Data Distribution and Probability Plots: Count
fig,ax = plt.subplots(ncols=2,nrows=3)
fig.set_size_inches(15,15)
sns.distplot(model['Count'], ax=ax[0][0])
stats.probplot(model["Count"], dist='norm', fit=True, plot=ax[0][1])
sns.distplot(stats.boxcox(model['Count'], 0), ax=ax[1][0])
stats.probplot(stats.boxcox(model["Count"], .5), dist='norm', fit=True, plot=ax[1][1])
sns.distplot(np.cbrt(model['Count']), ax=ax[2][0])
stats.probplot(np.cbrt(model["Count"]), dist='norm', fit=True, plot=ax[2][1])


# In[12]:


#Data Distribution and Probability Plots: Total Cost
fig,ax = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(15,15)
sns.distplot(model['total_cost'], ax=ax[0][0])
stats.probplot(model["total_cost"], dist='norm', fit=True, plot=ax[0][1])
sns.distplot(np.cbrt(model['total_cost']), ax=ax[1][0])
stats.probplot(np.cbrt(model["total_cost"]), dist='norm', fit=True, plot=ax[1][1])


# In[13]:


#Data Distribution and Probability Plots: Time Ridden
fig,ax = plt.subplots(ncols=2,nrows=3)
fig.set_size_inches(15,15)
sns.distplot(model['Time_Ridden'], ax=ax[0][0])
stats.probplot(model["Time_Ridden"], dist='norm', fit=True, plot=ax[0][1])
sns.distplot(stats.boxcox(model['Time_Ridden'], 0), ax=ax[1][0])
stats.probplot(stats.boxcox(model["Time_Ridden"], .5), dist='norm', fit=True, plot=ax[1][1])
sns.distplot(np.cbrt(model['Time_Ridden']), ax=ax[2][0])
stats.probplot(np.cbrt(model["Time_Ridden"]), dist='norm', fit=True, plot=ax[2][1])


# In[14]:


#Count boxplots on hour day month and season
fig, ax=plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(15,15)
sns.boxplot(data=model, y='Count', x='season', orient='v', ax=ax[0][0])
sns.boxplot(data=model, y='Count', x='month', orient='v', ax=ax[0][1])
sns.boxplot(data=model, y='Count', x='Day_Name', orient='v', ax=ax[1][0])
sns.boxplot(data=model, y='Count', x='hour', orient='v', ax=ax[1][1])


# In[15]:


#Time ridden boxplots of hour, day, month, and season
fig, ax=plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(15,15)
sns.boxplot(data=model, y='Time_Ridden', x='season', orient='v', ax=ax[0][0])
sns.boxplot(data=model, y='Time_Ridden', x='month', orient='v', ax=ax[0][1])
sns.boxplot(data=model, y='Time_Ridden', x='Day_Name', orient='v', ax=ax[1][0])
sns.boxplot(data=model, y='Time_Ridden', x='hour', orient='v', ax=ax[1][1])


# In[16]:


#Correlation Heatmap
cormat= model[:].corr()
mask = np.array(cormat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(10,10)
sns.heatmap(data=cormat, cmap="BuPu", vmax=0.5, mask=mask, square=True, annot=True, cbar=True)


# In[8]:


#Series plot with zero values
series = model.groupby([pd.Grouper(key='Time', freq='W')])['Count'].sum()
ax = series.plot(x='Time', y='Count', figsize = (10,8))
ax.set_title('Ride Count Over Time')
ax.set_ylabel('Count')


# In[9]:


#Series without null values
series[series == 0] = np.nan
series = series.dropna()
ax = series.plot(x='Time', y='Count', figsize = (10,8))
ax.set_title('Ride Count Over Time')
ax.set_ylabel('Count')


# In[ ]:


#export to be cleaned
series.to_csv(r'E:\TAMIDS 2019\Data\series.csv')


# In[ ]:





# In[ ]:


###TIME SERIES FORECASTING###


# In[ ]:





# In[3]:


#import cleaned data
series = pd.read_csv(r"E:\TAMIDS 2019\Data\timecorrected.csv", low_memory=False, parse_dates=['Time'])


# In[ ]:


#Series plot with no month before July 2016
ax = series.plot(x='Time', y='Count', figsize = (10,8))
ax.set_title('Ride Count Over Time')
ax.set_ylabel('Count')


# In[12]:


#Split data into training and testing 
train = series[0:650]
test = series[650:]

#visualize training and testing data
train.Count.plot(figsize=[12,8], title='Train Test Split on Time Series', fontsize=14)
test.Count.plot(figsize=[12,8], fontsize=14)
plt.xlabel('Days of Operation')
plt.ylabel('Daily Ride Count')
plt.legend(['Training Data', 'Testing Data'], loc='best')
plt.show


# In[6]:


#Naive Forecast
x = np.asarray(train.Count)
y_hat = test.copy()
y_hat['naive'] = x[len(x)-1]

plt.figure(figsize=(16,8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index, test['Count'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show


# In[7]:


#Get the RMSE of the naive forecast
rms_naive = sqrt(mean_squared_error(test.Count, y_hat.naive))
print(rms_naive)


# In[14]:


#Seasonal Decomposition
import statsmodels.api as sm
#Create Datetime Indices
train.index = pd.DatetimeIndex(freq='d', start=0, periods=len(train))
test.index = pd.DatetimeIndex(freq='d', start=0, periods=len(test))
#Train data
sm.tsa.seasonal_decompose(train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
plt.show
#Test data
sm.tsa.seasonal_decompose(test.Count).plot()
result = sm.tsa.stattools.adfuller(test.Count)
plt.show


# In[ ]:


#Exponential Smoothing
from statsmodels.tsa.api import ExponentialSmoothing, Holt
y_hat_avg = test.copy()

fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level = 0.6,smoothing_slope = 0.35)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
fit3 = Holt(np.asarray(train['Count']), damped=True).fit(smoothing_level = 0.8,smoothing_slope = 0.35)
y_hat_avg['Damped'] = fit3.forecast(len(test))

plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label="Holt's Linear trend")
plt.plot(y_hat_avg['Damped'], label="Additive damped trend")
plt.legend(loc='best')
plt.show()


# In[16]:


#load data
def GetData(fileName):
    return pd.read_csv(fileName, header=0, parse_dates=[0], index_col=0).values
data = GetData(r'E:\TAMIDS 2019\Data\timecorrected.csv')


# In[17]:


from statsmodels.tsa.arima_model import ARIMA

#define ARIMA model
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


# In[18]:


#creating training data
number = len(data)
trainsize = int(number *  0.7)
train = data[0:trainsize]
test= data[trainsize:number]
history = [x for x in train]


# In[19]:


#create predictions and begin forecasting
pred = list()
for time in range(len(test)):
    val = test[time]
    #forecast value
    prediction = StartARIMAForecasting(history, 1,1,2)
    print('history=%f, Predicted=%f' % (val, prediction))
    #add it in the list
    pred.append(prediction)
    history.append(val)


# In[22]:


#test metric
rmse = sqrt(mean_squared_error(test, pred))
rmse = rmse/23
print('Test RMSE: %.3f' % rmse)


# In[21]:


#plot the results
plt.plot(test)
plt.plot(pred, color='red')


# In[ ]:


# evaluate an ARIMA model for a given order (p,d,q)
# function from https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
# Credit to Jason Brownlee
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
 
# load dataset
series = Series.from_csv(r'E:\TAMIDS 2019\Data\timecorrected.csv', header=0)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# In[ ]:





# In[ ]:


###REGRESSION###


# In[ ]:





# In[ ]:


#Import Data and remove outliers
model = pd.read_csv(r"E:\TAMIDS 2019\Data\pred.csv", low_memory=False, parse_dates=["Time"])
model = model[np.abs(model["Count"]-model["Count"].mean())<=(model["Count"].std()*3)]
#remove unneeded columns
df = model.drop(model.columns[[0,9,10,11,12,14,15]], axis=1)
#One hot encode categorical variables
df = pd.get_dummies(data=df, columns=['Day_Name', 'month', 'season', 'weather_main'])
#Cylically encode hour variable
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)
#visualize cyclical encoding
ax = df.plot.scatter('hour_sin', 'hour_cos').set_aspect('equal')
#drop hour variable
df = df.drop(df.columns[[0]], axis=1)
#export to be cleaned and normalized in excel
df.to_csv(r'E:\TAMIDS 2019\Data\toclean.csv')


# In[ ]:


#import cleaned dataframe
df = pd.read_csv(r"E:\TAMIDS 2019\Data\dataframe.csv")
x = df.drop('Count', axis=1)
y = df.Count
#train test split
#80% training data 20% testing data, random state=1234
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1234)


# In[ ]:


##Linear Model##
from sklearn import linear_model

np.random.seed(1234)
ln = linear_model.SGDRegressor(penalty='elasticnet', max_iter=1000) 
ln.fit(x_train, y_train)

ln.predict(x_test)
ln.score(x_test, y_test)

## R^2 = 0.5472518033540035


# In[ ]:


##Random Forest##
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000, random_state=1234)
rf.fit(x_train, y_train)

rf.predict(x_test)
rf.score(x_train, y_train)
rf.score(x_test, y_test)

## R^2 = 0.7276071550831076


# In[ ]:


##Tuning Hyperparameters##
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#create base model
rf1 = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf1, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1234, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)


# In[ ]:


#return optimal parameters
print(rf_random.best_params_)
y_pred = rf_random.predict(x_test)
rf_random.score(x_test, y_test)

## R^2 = 0.7280520613504812


# In[ ]:





# In[ ]:





# In[ ]:




