#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import h5py

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Masking, add, Input, concatenate
from tensorflow.keras.models import Sequential, Model

from custom_mdn import MDN, get_custom_mixture_loss_func


def get_lstm(amt, params):
    return [LSTM(**params) for i in range(amt)]


def define_model2(N, batch_size, time_steps, vector_size, num_mixtures):
    "+ skip connections"
    lstm_params = {'units': N,
                   'activation': 'tanh',
                   'return_sequences': True,
                   'batch_input_shape': (batch_size, time_steps, vector_size)
    }
    enter = Input(batch_shape=(batch_size, time_steps, vector_size))
    # mask = Masking(mask_value=[0, 0, 0],
    #                batch_input_shape=(batch_size,
    #                time_steps, vector_size))(enter)
    raw_lstm1, raw_lstm2, raw_lstm3 = get_lstm(3, lstm_params)
    input_proxy = Dense(N)(enter)

    lstm1 = raw_lstm1(enter)
    lvl1_out = add([input_proxy, lstm1])
    lstm2 = raw_lstm2(lvl1_out)
    lvl2_out = add([input_proxy, lstm2])
    lstm3 = raw_lstm3(lvl2_out)
    lvl3_out = add([input_proxy, lstm3])

    out_proxy = Dense(vector_size)
    lstm1_proxy = out_proxy(lstm1)
    lstm2_proxy = out_proxy(lstm2)
    lstm3_proxy = out_proxy(lstm3)

    out_dense = Dense(units=vector_size, activation='linear')(lvl3_out)
    out_proxy = add([out_dense, lstm1_proxy, lstm2_proxy, lstm3_proxy])
    out = MDN(vector_size, num_mixtures)(out_proxy)

    m = Model(inputs=enter, outputs=out)
    m.compile(optimizer='rmsprop',
              loss=get_custom_mixture_loss_func(num_mixtures))
    return m


if __name__ == '__main__':
    N = 10
    with h5py.File('../dataset.h5', 'r') as f:
        X = f['lines'][:]
    X = X[:200]
    batch_size = 32
    _, time_steps, vector_size = X.shape
    m = define_model2(N, None, None, vector_size, 5)
    print(m.summary())
    size = X.shape[0] - X.shape[0] % batch_size
    X_train = X[:size, :-1, :]
    Y_train = X[:size, 1:, :]
    X_train = tf.convert_to_tensor(X_train.astype(np.float))
    Y_train = tf.convert_to_tensor(Y_train.astype(np.float))
    m.fit(X_train, Y_train,
          batch_size=None, epochs=1)
    m.save('hwg_model3.h5')
