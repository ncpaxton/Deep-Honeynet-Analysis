from keras.models import Sequential
from keras.layers import  LSTM, Dropout, Dense, Activation, TimeDistributed

"""
Building the base LSTM architecture
"""
def build_lstm_model(use_dropout, LSTM_size, Dense_size, num_features, hyperparameters):
    num_steps = hyperparameters['num_steps']
    num_classes = hyperparameters['num_classes']
    model = Sequential()
    model.add(LSTM(LSTM_size, activation='relu', return_sequences=True, input_shape=(num_steps,num_features)))
    model.add(LSTM(LSTM_size, activation='relu', return_sequences=True))
    model.add(Dense(Dense_size, activation='relu'))
    if (use_dropout):
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_classes)))
    model.add(Activation('softmax'))
    return model
