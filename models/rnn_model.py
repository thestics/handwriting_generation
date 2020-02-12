#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, LSTM, Layer, TimeDistributed
from tensorflow.keras.models import Model, Sequential

from batch_generator import BatchGenerator


def define_model(N):
    m = Sequential()
    m.add(LSTM(units=N, activation='relu', return_sequences=True, input_shape=(None, 3)))
    m.add(Dense(3, activation='linear'))
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m
    # x = Input(shape=(None, 3))
    # lstm1 = LSTM(units=N, activation='relu', return_sequences=True)(x)
    # # lstm2 = LSTM(units=N, activation='relu', return_sequences=True)(lstm1)
    # # lstm3 = LSTM(units=N, activation='relu', return_sequences=True)(lstm2)
    # out = Dense(3, activation='linear')(lstm3)
    #
    # m = Model(inputs=x, outputs=out)
    # m.compile(optimizer='adam', loss='categorical_crossentropy',
    #           metrics=['accuracy'])
    # return m


if __name__ == '__main__':
    N = 10
    batch_size = 1053

    # size = 10001
    m = define_model(N)
    X = np.load('../lines.npy', allow_pickle=True)
    X_train = X[:, :-1, :]
    Y_train = X[:, 1, :]
    m.build()
    m.summary()
    m.fit(X_train, Y_train)
