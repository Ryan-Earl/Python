#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math 
import sklearn
import keras
import tensorflow
import pylab
import calendar 
import matplotlib
import scipy as sp
from scipy import stats
import statsmodels.api as sm
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
style.use('seaborn-colorblind')


# In[ ]:


#seperate data by region
dtla = station.where(station['Region'] == 'DTLA')
dtla = dtla[np.isfinite(dtla['count'])]
#group by time
dtla_count = dtla.groupby(['joinkey'])["count"].sum()
#export to clean in excel
dtla_count.to_csv(r'E:\TAMIDS 2019\Data\dtla_count.csv')
#import clean csv
dtla_count = pd.read_csv('E:\TAMIDS 2019\Data\dtla_correct.csv', infer_datetime_format=True, parse_dates=True)
dtla_count = dtla_count[np.abs(dtla_count['ride_count']-dtla_count['ride_count'].mean())<=(dtla_count['ride_count'].std()*3)]
dtla_count['time'] =  pd.to_datetime(dtla_count['time'])
dtla_count = dtla_count.groupby([pd.Grouper(key='time', freq='D')]).sum()
dtla_count[dtla_count == 0] = np.nan
dtla_count = dtla_count.dropna()
dtla_count.to_csv(r'E:\TAMIDS 2019\Data\dtla.csv')
dtla_count = pd.read_csv('E:\TAMIDS 2019\Data\dtla1.csv', infer_datetime_format=True, parse_dates=['time'])
#Series plot with no month before July 2016
ax = dtla_count.plot(x='time', y='ride_count', figsize = (12,8))
ax.set_title('Ride Count Over Time in DTLA')
ax.set_ylabel('Count')
ax.set_xlabel('Time')


# In[2]:


train = pd.read_csv('E:\TAMIDS 2019\Data\wtfff.csv', infer_datetime_format=True, parse_dates=['time'])
test = pd.read_csv('E:\TAMIDS 2019\Data\wtff.csv', infer_datetime_format=True, parse_dates=['time'])


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(train['time'], train['ride_count'], label='Train')
plt.plot(test['time'], test['ride_count'], label='Test')
plt.legend(loc='best')
plt.xlabel("Date")
plt.ylabel("Ride Count")
plt.title("Train and Test Split DTLA")
plt.show


# In[3]:


#set index as Time and drop column
train = train.set_index(['time'], drop=True)
test = test.set_index(['time'], drop=True)

#normalize train and test data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
train_dtla = scaler.fit_transform(train)
test_dtla = scaler.fit_transform(test)
train_sc_df = pd.DataFrame(train_dtla, columns=['Y'], index=train.index)
test_sc_df = pd.DataFrame(test_dtla, columns=['Y'], index=test.index)


for i in range(1,2):
    train_sc_df['X_{}'.format(i)] = train_sc_df['Y'].shift(i)
    test_sc_df['X_{}'.format(i)] = test_sc_df['Y'].shift(i)

X_train = train_sc_df.dropna().drop('Y', axis=1)
y_train = train_sc_df.dropna().drop('X_1', axis=1)
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

X_test = test_sc_df.dropna().drop('Y', axis=1)
y_test = test_sc_df.dropna().drop('X_1', axis=1)
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print('Train shape: ', X_train_lstm.shape)
print('Test shape: ', X_test_lstm.shape)


# In[4]:


nn_model = Sequential()
nn_model.add(Dense(12, input_dim=1, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)


# In[5]:


y_pred_test_nn = scaler.inverse_transform(nn_model.predict(X_test))
y_train_pred_nn = scaler.inverse_transform(nn_model.predict(X_train))
orig_train = scaler.inverse_transform(y_train)
orig_test = scaler.inverse_transform(y_test)
print(sqrt(mean_squared_error(orig_train, y_train_pred_nn)))
print(sqrt(mean_squared_error(orig_test, y_pred_test_nn)))


# In[6]:


lstm_model = Sequential()
lstm_model.add(LSTM(12, input_shape=(1, X_train_lstm.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm = lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])


# In[7]:


y_pred_test_lstm = scaler.inverse_transform(lstm_model.predict(X_test_lstm))
y_train_pred_lstm = scaler.inverse_transform(lstm_model.predict(X_train_lstm))
print(sqrt(mean_squared_error(orig_train, y_train_pred_lstm)))
print(sqrt(mean_squared_error(orig_test, y_pred_test_lstm)))


# In[8]:


plt.figure(figsize=(10, 6))
plt.plot()
plt.plot(orig_test, label='True Count')
plt.plot(y_pred_test_nn, label='NN')
plt.plot(y_pred_test_lstm, label='LSTM')
plt.title("Model Performance DTLA")
plt.xlabel('Observation')
plt.ylabel('Ride Count')
plt.legend(loc='best')
plt.show();

