
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import ipaddress
from keras.utils import normalize


#drop columns we dont need or have 1 unique value
def drop_cols(df):
    drop_list = ['Packet_Number','IP_Address_Norm','DF','MF','Day','Month','Hour','Day_of_Year']
    for col in drop_list:
        df = df.drop(col, axis=1)
    return df

#convert categorical features into one-hot vector
def to_onehot(df):
    feature_list = []
    for col in df.columns:
        print(col + ": " + str(len(df[col].unique())))
        if(len(df[col].unique()) < 20 and col != 'Category'):
            feature_list.append(col)
    
    for col in feature_list:
        df = pd.concat([df,pd.get_dummies(df[col], prefix=col)], axis=1)
        df = df.drop(col, axis=1)

    return df

"""
IP Address Preprocessing
"""
def convert_IPs(df, encoder):
    if (encoder):
        """
        Instead of parsing IPs, we can encode them as binary inputs and still capture the
        underlying meaning for each IPs subsections.
        """
        columns = ['Bit' + str(i) for i in range(32,0,-1)]
        df['IP_Address'] = pd.Series([list(format(int(ipaddress.IPv4Address(row)),'032b')) for row in df['IP_Address']])
        df[columns] = pd.DataFrame(df['IP_Address'].values.tolist())
        for col in columns:
            df[col] = pd.to_numeric(df[col])
       
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

    df = df.drop('IP_Address', axis=1)
    return df

"""
Label Encoding
"""
def convert_label(df):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df['Category'])
    df['Category'] = encoder.transform(df['Category'])
    return df

"""
Normalize features
"""
def normalize_features(df, use_keras_normalizer, maxMin_scaler):
    #norm_columns = []
    #exclude_lst = ['Category','IP_Address','TTL']
    idx = df.columns.get_loc('Category')
    norm_columns = df.columns.tolist()
    norm_columns = norm_columns[:idx]
    #for col in df.columns:
    #    if (col != 'Category'):
    #        norm_columns.append(col)

    if(use_keras_normalizer):
      None                                                              #ignoring it for now
    elif (maxMin_scaler):
        for col in norm_columns:
            min_max_scaler = preprocessing.MinMaxScaler()               #create a scaler instance
            float_array = df[col].values.astype(float).reshape(-1,1)
            scaled_array = min_max_scaler.fit_transform(float_array)
            df[col] = pd.DataFrame(scaled_array)
    else:
        for col in norm_columns:
            min_max_scaler = preprocessing.StandardScaler()             #create a scaler instance
            float_array = df[col].values.astype(float).reshape(-1,1)
            scaled_array = min_max_scaler.fit_transform(float_array)
            df[col] = pd.DataFrame(scaled_array)

    return df


#load data
training = pd.read_csv('all_training.csv')
training = training.drop('Unnamed: 0', axis=1)
dev = pd.read_csv('all_dev.csv')
dev = dev.drop('Unnamed: 0', axis=1)
test = pd.read_csv('all_test.csv')
test = test.drop('Unnamed: 0', axis=1)

#combine all in one table
frames = [training, dev, test]
combined = pd.concat(frames)

#drop few columns, convert to one-hot and convert IPs
combined = drop_cols(combined)
combined = to_onehot(combined)
combined = convert_IPs(combined, True)  

#retrieve train/dev/test sets back
new_training = combined[0:training.shape[0]]
new_dev = combined[training.shape[0]:training.shape[0] + dev.shape[0]]
new_test = combined[training.shape[0] + dev.shape[0]: ]

#finish rest of the preprocessing and save the final versions
frames = [new_training, new_dev, new_test]
data_lst = ['training','dev','test']

for df,title in zip(frames,data_lst):
    #sort by the current_time
    #df = df.sort_values(by=['Current_Time'])
    #randomly sample
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop('Current_Time', axis=1)
    df = normalize_features(df, False,False)
    df = convert_label(df)
    
    #finally rearrange the columns
    idx = df.columns.get_loc('Category')
    cols = df.columns.tolist()
    cols = cols[:idx] + cols[idx+1:] + cols[idx:idx+1]
    df = df[cols]
    df.to_csv('all_' + title + '_processed.csv')
    
    


####################################
#####      PREVIOUS WORK      ######
####################################
"""    
#create train,validation,test sets
    row_num = len(df.index)
    train_size = round(row_num * 0.8)
    validation_size = (row_num - train_size) // 2
    df_train = df[:train_size]
    df_val = df[train_size: train_size + validation_size]
    df_test = df[train_size + validation_size : ]
    df_train.to_csv('all_train.csv')
    df_val.to_csv('all_val.csv')
    df_train.to_csv('all_test.csv')    
    
"""    

"""
##if we want to convert to a Tensor
#Code starts here:
inputs = tf.convert_to_tensor(df_new.values, dtype=np.float32)
with tf.Session() as sess:
    print (sess.run(inputs))
"""

"""
Parse IP address into columns
#Code starts here:
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
#Code starts here:
df.insert(1,'Microseconds', datetimeSeries.dt.microsecond)
df.insert(1,'Second', datetimeSeries.dt.second)
df.insert(1,'Minute', datetimeSeries.dt.minute)
df.insert(1,'Day', datetimeSeries.dt.day)
df.insert(1,'Week', datetimeSeries.dt.week)
df.insert(1,'Day_Of_Week', datetimeSeries.dt.dayofweek)
df.insert(1,'Month', datetimeSeries.dt.month)
"""

"""
Extract only the "hour" component from the timestamp in order to group incoming
packets by the hour
#Code starts here:
datetimeSeries = pd.to_datetime(df['Current_Time'], unit='s')
minutetimeSeries = datetimeSeries.dt.minute
df.insert(1,'Minute_v2', datetimeSeries.dt.minute)
"""