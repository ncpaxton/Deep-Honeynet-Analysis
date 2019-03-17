#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:49:00 2019

@author: berkaypolat
"""
from keras.models import Sequential
from keras.layers import  GRU, Dropout, Dense, Activation, TimeDistributed

"""
Building a RNN architecture with GRU
recurrent_activation's defualt value is 'hard_sigmoid'
"""
def build_bilstm_model(use_dropout, GRU_size, Dense_size, num_steps, num_features, num_classes, recurrent_activation='hard_sigmoid'):
    model = Sequential()
    model.add(GRU(GRU_size, activation='relu', return_sequences=True, input_shape=(num_steps,num_features), recurrent_activation=recurrent_activation))
    model.add(GRU(GRU_size, activation='relu', return_sequences=True, input_shape=(num_steps,num_features), recurrent_activation=recurrent_activation));
    model.add(Dense(Dense_size, activation='relu'))
    if (use_dropout):
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    return model
