#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:29:00 2019

@author: berkaypolat
"""
from keras.models import Sequential
from keras.layers import  LSTM, Dropout, Dense, Activation, TimeDistributed, Bidirectional

"""
Building the Bidirectional LSTM architecture
merge_mode can be one of the options: 'sum', 'mul', 'concat', 'ave'
"""
def build_bilstm_model(use_dropout, LSTM_size, Dense_size, num_steps, num_features, num_classes, merge_mode):
    model = Sequential()
    model.add(Bidirectional(LSTM(LSTM_size, activation='relu', return_sequences=True, input_shape=(num_steps,num_features)), merge_mode=merge_mode))
    model.add(Bidirectional(LSTM(LSTM_size, activation='relu', return_sequences=True), merge_mode=merge_mode))
    model.add(Dense(Dense_size, activation='relu'))
    if (use_dropout):
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    return model
