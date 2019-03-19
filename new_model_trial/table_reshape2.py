#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:08:03 2019

@author: berkaypolat
"""

import pandas as pd

data_lst = ['training','dev','test']

#combine IoT data
for item in data_lst:
    df = pd.read_csv("iot_traffic/iot_" + item + ".csv")
    df2 = pd.read_csv("iot_traffic/extra_iot_" + item + ".csv")
    frames = [df, df2]
    new_df = pd.concat(frames)
    new_df.to_csv("iot_traffic/all_iot_" + item + ".csv")


#combine web traffic data
for item in data_lst:
    df = pd.read_csv("web_traffic/web_" + item + ".csv")
    df2 = pd.read_csv("web_traffic/extra_web_" + item + ".csv")
    frames = [df, df2]
    new_df = pd.concat(frames)
    new_df.to_csv("web_traffic/all_web_" + item + ".csv")    

#create 3 main Train/Validation/Test sets
for item in data_lst:
    normal = pd.read_csv("normal_traffic/normal_traffic_" + item + ".csv")
    iot = pd.read_csv("iot_traffic/all_iot_" + item + ".csv")
    iot = iot.drop('Unnamed: 0', axis=1)
    
    normal = normal[:iot.shape[0]]
    web = pd.read_csv("web_traffic/all_web_" + item + ".csv")
    web = web.drop('Unnamed: 0', axis=1)
    frames = [normal, iot, web]
    new_df = pd.concat(frames)
    new_df.to_csv("all_" + item + ".csv")

