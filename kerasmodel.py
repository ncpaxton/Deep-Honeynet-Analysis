#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:52:36 2019

@author: berkaypolat
"""

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import  LSTM, Dropout, Dense, Activation, TimeDistributed
from keras.callbacks import ModelCheckpoint



"""
Creating a data generator class in order to feed in an
iterator object for the final model
"""
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, num_classes, num_features):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.current_idx = 0
        self.skip_step = num_steps
        self.num_features = num_features

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, self.num_features))
        y = np.zeros((self.batch_size, self.num_steps, self.num_classes))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0

                x[i, :, :] = self.data[self.current_idx : self.current_idx + self.num_steps, 0:self.num_features]
                temp_y = self.data[self.current_idx:self.current_idx + self.num_steps, -1]
                y[i, : , :] = to_categorical(temp_y, num_classes = self.num_classes)
                self.current_idx += self.skip_step

            yield x,y

"""
Retrieve hyperparameters
"""
def get_hyperparameters():
    batch_size = 16
    num_classes = 3
    num_steps = 25
    return batch_size, num_classes, num_steps

"""
Loading the data and extracting useful information
"""
def load_train_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)

    return df.values, len(df.columns)-1

def load_validation_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    return df.values

"""
Building the LSTM architecture
"""
def build_model(use_dropout, LSTM_size, Dense_size, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(LSTM(LSTM_size, activation='relu', return_sequences=True, input_shape=(num_steps,num_features)))
    model.add(LSTM(LSTM_size, activation='relu', return_sequences=True))
    model.add(Dense(Dense_size, activation='relu'))
    if (use_dropout):
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    return model

"""
Run and Compile the LSTM NN
"""
def run_model(model,train_data,valid_data, train_data_generator, valid_data_generator, batch_size, num_steps):
    num_epochs = 10
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    #can be convert to True for saving best model
    checkpoints = ModelCheckpoint(filepath='model_history/model-{epoch:02d}.hdf5', save_best_only=False, verbose=1)
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data= valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpoints])

"""
NOTE !!!
Something with the current version of Keras in conda environment has been causing an error.
Running this command should be able to make it work
pip uninstall keras
conda install -c conda-forge keras==2.1.2
"""

batch_size, num_classes, num_steps = get_hyperparameters()
train_data, num_features = load_train_data('all_train.csv')
validation_data = load_validation_data('all_val.csv')
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, num_classes, num_features)
validation_data_generator = KerasBatchGenerator(validation_data, num_steps, batch_size, num_classes, num_features)
model = build_model(True, 200, 100, num_steps, num_features, num_classes)
run_model(model,train_data,validation_data,train_data_generator, validation_data_generator, batch_size, num_steps)
