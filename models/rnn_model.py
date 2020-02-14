#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import h5py

import numpy as np

from tensorflow.keras.layers import Dense, LSTM, Masking, add, Input, concatenate
from tensorflow.keras.models import Sequential, Model


def define_model(N, batch_size, time_steps, vector_size):
    m = Sequential()
    m.add(Masking(mask_value=[0, 0, 0],
                  batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(LSTM(units=N, activation='tanh', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(LSTM(units=N, activation='tanh', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(LSTM(units=N, activation='tanh', return_sequences=True,
               batch_input_shape=(batch_size, time_steps, vector_size)))
    m.add(Dense(units=vector_size, activation='linear'))
    m.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return m


def get_lstm(amt, params):
    return [LSTM(**params) for i in range(amt)]


def define_model2(N, batch_size, time_steps, vector_size):
    "+ skip connections"
    lstm_params = {'units': N,
                   'activation': 'tanh',
                   'return_sequences': True,
                   'batch_input_shape': (batch_size, time_steps, vector_size)
    }
    enter = Input(batch_shape=(batch_size, time_steps, vector_size))
    mask = Masking(mask_value=[0, 0, 0],
                   batch_input_shape=(batch_size, time_steps, vector_size))(enter)
    raw_lstm1, raw_lstm2, raw_lstm3 = get_lstm(3, lstm_params)
    lstm1 = raw_lstm1(mask)
    lvl1_out = concatenate([mask, lstm1])
    lstm2 = raw_lstm2(lvl1_out)
    lvl2_out = concatenate([lstm2, mask])
    lstm3 = raw_lstm3(lvl2_out)
    lvl3_out = concatenate([lstm3, mask])

    out = Dense(units=vector_size, activation='linear')(lvl3_out)
    # out = add([raw_out, lstm1, lstm2, lstm3])
    m = Model(inputs=enter, outputs=out)
    m.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    return m



if __name__ == '__main__':
    N = 10
    with h5py.File('../dataset.h5', 'r') as f:
        X = f['lines'][:]
    batch_size = 32
    _, time_steps, vector_size = X.shape
    m = define_model2(N, batch_size, time_steps - 1, vector_size)
    # m.build()
    print(m.summary())
    size = X.shape[0] - X.shape[0] % batch_size
    X_train = X[:size, :-1, :]
    Y_train = X[:size, 1:, :]
    m.fit(X_train, Y_train, batch_size=batch_size, epochs=10)
    m.save('hwg_model.h5')
