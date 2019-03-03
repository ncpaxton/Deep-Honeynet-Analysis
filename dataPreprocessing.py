
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

df = pd.read_csv('model_features.csv')

"""
Parse timestamp into different time values
"""
datetimeSeries = pd.to_datetime(df['Current_Time'], unit='s')
df.insert(1,'Microseconds', datetimeSeries.dt.microsecond)
df.insert(1,'Second', datetimeSeries.dt.second)
df.insert(1,'Minute', datetimeSeries.dt.minute)
df.insert(1,'Hour', datetimeSeries.dt.hour)
df.insert(1,'Day', datetimeSeries.dt.day)
df.insert(1,'Week', datetimeSeries.dt.week)
df.insert(1,'Day_Of_Week', datetimeSeries.dt.dayofweek)
df.insert(1,'Month', datetimeSeries.dt.month)
df = df.drop('Current_Time', axis=1)
df.head()

"""
Parse IP address into columns
"""
ip_classes = {}                                     #maps each ip part to number of unique values it contains
iPs = df['IP_Address'].str.split('.', expand=True)
index = df.columns.get_loc('IP_Address') + 1
for  i in range(iPs.shape[1]):
    df.insert(i+index,'IPSec' + str(i+1), iPs[i])
    ip_classes['n' + str(i+1) + '_category'] =  df['IPSec' + str(i+1)].unique().size
df.head()

"""
We can either convert them to one-hot or categorical encoding
"""
one_hot = False;
cols = ['IPSec1','IPSec2','IPSec3','IPSec4']
if (one_hot):
    #do one-hot encoding for each part of the IP numbers
    df = pd.concat([df,pd.get_dummies(df[['IPSec1','IPSec2','IPSec3','IPSec4']])], axis=1)
else:
    #do categorical encoding for each part of the IP numbers
    for col in cols:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(df[col])
        df[col] = encoder.transform(df[col])
df.head()


"""
Normalize microseconds, packet length and packet_checksum columns
"""
columns = ['Microseconds', 'Packet_Length', 'Packet_Checksum']
for col in columns:
    min_max_scaler = preprocessing.MinMaxScaler()               #create a scaler instance
    float_array = df[col].values.astype(float).reshape(-1,1)
    scaled_array = min_max_scaler.fit_transform(float_array)
    df[col] = pd.DataFrame(scaled_array)


df = df.drop('IP_Address', axis=1)
df.head()
df.to_csv('new_model_features.csv')


##if we want to convert to a Tensor
inputs = tf.convert_to_tensor(df_new.values, dtype=np.float32)
with tf.Session() as sess:
    print (sess.run(inputs))

