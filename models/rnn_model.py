#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import numpy as np
from tensorflow.keras.layers import Dense, Input, LSTM, Layer, TimeDistributed
from tensorflow.keras.models import Model, Sequential


def define_model(N, batch_size, time_steps, vector_size):
    m = Sequential()
    m.add(LSTM(units=N, activation='relu', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(LSTM(units=N, activation='relu', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(LSTM(units=N, activation='relu', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(Dense(units=vector_size, activation='linear'))
    m.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return m


if __name__ == '__main__':
    N = 40
    X = np.load('../lines.npy', allow_pickle=True)
    batch_size = 32
    _, time_steps, vector_size = X.shape
    m = define_model(N, 32, time_steps - 1, vector_size)
    size = X.shape[0] - X.shape[0] % batch_size
    X_train = X[:size, :-1, :]
    Y_train = X[:size, 1:, :]
    m.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
