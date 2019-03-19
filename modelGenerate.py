#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:31:00 2019

This file compiles and fits all the different models of RNN implemented for this project (except
dynamic-RNN).
Model results are written to a .hdf5 formatted files and can be loaded_back for additional
analysis.
NOTE !!!
Something with the current version of Keras in conda environment has been causing an error.
You should install the Keras version 2.1.2. Run the the following commands on your terminal
pip uninstall keras
conda install -c conda-forge keras==2.1.2

@author: berkaypolat
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import  LSTM, Dropout, Dense, Activation, TimeDistributed
from keras.callbacks import ModelCheckpoint
from bilstm import build_bilstm_model
from grumodel import build_gru_model
from lstm import build_lstm_model

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

def load_test_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    return df.values

"""
Retrieve hyperparameters
"""
def get_hyperparameters():
    return {'batch_size': 16, 'num_classes':3, 'num_steps':10, 'num_epochs': 200}

"""
Creating a data generator class in order to feed in an
iterator object for the final model
"""
class KerasBatchGenerator(object):

    def __init__(self, data, hyperparameters, num_features):
        self.data = data
        self.num_steps = hyperparameters['num_steps']
        self.batch_size = hyperparameters['batch_size']
        self.num_classes = hyperparameters['num_classes']
        self.current_idx = 0
        if(self.batch_size == 1):
            self.skip_step = 1
        else:
            self.skip_step = hyperparameters['num_steps']
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
Run and Compile the LSTM NN
"""
def run_model(model_name, model, data_length, generators, hyperparameters):
    train_len, valid_len = data_length['train'], data_length['validation']
    train_gen, valid_gen = generators['train_data_generator'], generators['validation_data_generator']
    batch_size, num_epochs = hyperparameters['batch_size'], hyperparameters['num_epochs']
    num_steps = hyperparameters['num_steps']
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    if (model_name == 'GRU'):
        filepath = 'gru_model_history/model-{epoch:02d}.hdf5'
    elif(model_name == 'BILSTM'):
        filepath = 'bilstm_model_history/model-{epoch:02d}.hdf5'
    else:
        filepath = 'lstm_model_history/model-{epoch:02d}.hdf5'
    #can be convert to True for saving best model
    checkpoints = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)
    model_history = model.fit_generator(train_gen.generate(), train_len//(batch_size*num_steps), num_epochs,
                        validation_data= valid_gen.generate(),
                        validation_steps= valid_len//(batch_size*num_steps), callbacks=[checkpoints])

    return model_history

"""
Generates and runs the models
"""
def call_models(model_lst):
    models_history = []
    hyperparameters = get_hyperparameters()
    train_data, num_features = load_train_data('all_training_processed.csv')
    validation_data = load_validation_data('all_dev_processed.csv')

    for model_name in model_lst:
        train_data_generator = KerasBatchGenerator(train_data, hyperparameters, num_features)
        validation_data_generator = KerasBatchGenerator(validation_data, hyperparameters, num_features)
        if(model_name == 'GRU'):
            model = build_gru_model(True, 200, 100, num_features, hyperparameters)
        elif(model_name == 'BILSTM'):
            model = build_bilstm_model(True, 200, 100, num_features, hyperparameters, merge_mode='concat')
        else:
            model = build_lstm_model(True,200,100,num_features, hyperparameters)

        data_length = {'train':len(train_data), 'validation':len(validation_data)}
        generators = {'train_data_generator': train_data_generator, 'validation_data_generator':validation_data_generator}
        models_history.append(run_model(model_name,model,data_length,generators, hyperparameters))

    return models_history

"""
Get train loss and validation loss values for each epoch
The History Object keys are:
['val_loss', 'val_categorical_accuracy', 'loss', 'categorical_accuracy']
"""
def get_loss_tables(history_lst, model_lst):
    loss_table = pd.DataFrame()
    val_loss_table = pd.DataFrame()
    for i in range(len(model_lst)):
        loss_table[model_lst[i]] = history_lst[i].history['loss']
        val_loss_table[model_lst[i]] = history_lst[i].history['val_loss']
    return loss_table, val_loss_table

if __name__ == '__main__':
    model_lst = ['LSTM', 'GRU', 'BILSTM']
    history_lst = call_models(model_lst)
    train_losses, val_losses = get_loss_tables(history_lst, model_lst)
    train_losses.to_csv("train_losses.csv")
    val_losses.to_csv("validation_losses.csv")
