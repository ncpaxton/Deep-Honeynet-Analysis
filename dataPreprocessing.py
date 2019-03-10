
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import ipaddress

df = pd.read_csv('model_features.csv')

"""
Extract only the "hour" component from the timestamp in order to group incoming
packets by the hour
"""
datetimeSeries = pd.to_datetime(df['Current_Time'], unit='s')
hourtimeSeries = datetimeSeries.dt.hour
df.insert(1,'Hour', datetimeSeries.dt.hour)
df = df.drop('Current_Time', axis=1)


encoder = True

if (encoder):
    """
    Instead of parsing IPs, we can encode them as binary inputs and still capture the
    underlying meaning for each IPs subsections.
    """
    columns = ['Bit' + str(i) for i in range(32,0,-1)]
    df['IP_Address'] = pd.Series([list(format(int(ipaddress.IPv4Address(row)),'032b')) for row in df['IP_Address']])
    df[columns] = pd.DataFrame(df['IP_Address'].values.tolist())
else:
    """
    Our second approach will be to convert them to one-hot vectors. Previously we
    went to categorical encoding as well. You can change the boolean value to choose
    between the two.
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


"""
Normalize packet length and packet_checksum columns. Additional features can
also be normalized here
"""
columns = ['Packet_Length', 'Packet_Checksum']
for col in columns:
    min_max_scaler = preprocessing.MinMaxScaler()               #create a scaler instance
    float_array = df[col].values.astype(float).reshape(-1,1)
    scaled_array = min_max_scaler.fit_transform(float_array)
    df[col] = pd.DataFrame(scaled_array)


cols = df.columns.tolist()
cols = cols[:5] + cols[6:] + cols[5:6]

df = df[cols]

df = df.drop('IP_Address', axis=1)
df.to_csv('new_model_features.csv')



####################################
#####      PREVIOUS WORK      ######
####################################

"""
##if we want to convert to a Tensor
inputs = tf.convert_to_tensor(df_new.values, dtype=np.float32)
with tf.Session() as sess:
    print (sess.run(inputs))
"""

"""
Parse IP address into columns

ip_classes = {}                                     #maps each ip part to number of unique values it contains
iPs = df['IP_Address'].str.split('.', expand=True)
index = df.columns.get_loc('IP_Address') + 1
for  i in range(iPs.shape[1]):
    df.insert(i+index,'IPSec' + str(i+1), iPs[i])
    ip_classes['n' + str(i+1) + '_category'] =  df['IPSec' + str(i+1)].unique().size
df.head()
"""

"""
Previous parsing method using the timestamp

df.insert(1,'Microseconds', datetimeSeries.dt.microsecond)
df.insert(1,'Second', datetimeSeries.dt.second)
df.insert(1,'Minute', datetimeSeries.dt.minute)
df.insert(1,'Day', datetimeSeries.dt.day)
df.insert(1,'Week', datetimeSeries.dt.week)
df.insert(1,'Day_Of_Week', datetimeSeries.dt.dayofweek)
df.insert(1,'Month', datetimeSeries.dt.month)
"""