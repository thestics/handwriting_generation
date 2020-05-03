#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import h5py
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Masking, add, Input, concatenate
from tensorflow.keras.models import Sequential, Model

from custom_mdn import MDN, get_mixture_loss_func as _get_mixture_loss_func, get_mixture_mse_accuracy, sample

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def get_lstm(amt, params):
    return [LSTM(**params) for i in range(amt)]


def define_model(N, depth, batch_size, time_steps, vector_size, num_mixtures):
    lstm_params = {'units': N,
                   'activation': 'tanh',
                   'return_sequences': True,
                   'batch_input_shape': (batch_size, time_steps, vector_size)
                   }
    model = Sequential()
    for i in range(depth):
        model.add(LSTM(**lstm_params))

    model.add(MDN(vector_size - 1, num_mixtures))
    model.compile(optimizer='rmsprop',
                  loss=_get_mixture_loss_func(vector_size - 1, num_mixtures))
    return model


def define_model2(N, batch_size, time_steps, vector_size, num_mixtures):
    "+ skip connections"
    lstm_params = {'units': N,
                   'activation': 'tanh',
                   'return_sequences': True,
                   'batch_input_shape': (batch_size, time_steps, vector_size)
    }
    enter = Input(batch_shape=(batch_size, time_steps, vector_size))
    # mask = Masking(mask_value=(0, 0, 0),
    #                batch_input_shape=(batch_size,time_steps, vector_size)
    #                )(enter)
    raw_lstm1, raw_lstm2, raw_lstm3 = get_lstm(3, lstm_params)
    input_proxy = Dense(N, activation='tanh')(enter)

    lstm1 = raw_lstm1(enter)
    lvl1_out = add([input_proxy, lstm1])
    lstm2 = raw_lstm2(lvl1_out)
    lvl2_out = add([input_proxy, lstm2])
    lstm3 = raw_lstm3(lvl2_out)
    lvl3_out = add([input_proxy, lstm3])

    out_proxy = Dense(vector_size, activation='tanh')
    lstm1_proxy = out_proxy(lstm1)
    lstm2_proxy = out_proxy(lstm2)
    lstm3_proxy = out_proxy(lstm3)

    out_dense = Dense(units=vector_size, activation='linear')(lvl3_out)
    out_proxy = add([out_dense, lstm1_proxy, lstm2_proxy, lstm3_proxy])
    out = MDN(vector_size - 1, num_mixtures)(out_proxy)

    m = Model(inputs=enter, outputs=out)
    m.compile(optimizer='rmsprop',
              loss=_get_mixture_loss_func(vector_size - 1, num_mixtures),
              # metrics=[get_mixture_mse_accuracy(vector_size - 1, num_mixtures), ]
              )
    return m


def train_model():
    lstm_units = 400
    num_mixtures = 20
    with h5py.File('../dataset.h5', 'r') as f:
        X = f['lines'][:]
    batch_size = 8
    _, time_steps, vector_size = X.shape
    m = define_model2(lstm_units, batch_size, time_steps - 1, vector_size,
                      num_mixtures)
    print(m.summary())
    size = X.shape[0] - X.shape[0] % batch_size
    X_train = X[:size, :-1, :]
    Y_train = X[:size, 1:, :]
    X_train = tf.convert_to_tensor(X_train.astype(np.float32))
    Y_train = tf.convert_to_tensor(Y_train.astype(np.float32))
    with tf.device("/gpu:0"):
        m.fit(X_train, Y_train,
              batch_size=batch_size, epochs=5)
    m.save_weights('model/hwg_model.tf', overwrite=True)


def build_sample(N):
    m = define_model2(400, 8, 1939, 3, 20)
    m.load_weights('model/hwg_model.tf')

    res = [c.numpy() for c in sample(m, N, 3, 20)]
    return res


def sample_to_series(s, x0, y0):
    res = []
    for c in s:
        x0 += c[1]
        y0 += c[0]
        res.append((x0, y0, c[2]))
    return res


def plot_sample(s):
    to_plotX = []
    to_plotY = []
    for x, y, stroke_end in s:
        if not stroke_end:
            to_plotX.append(x)
            to_plotY.append(y)
        else:
            plt.plot(to_plotX, to_plotY, color='b')
            to_plotX.clear()
            to_plotY.clear()



if __name__ == '__main__':
    sample = build_sample(100)
    sample = sample_to_series(sample, 0, 0)
    plot_sample(sample)
    plt.show()
