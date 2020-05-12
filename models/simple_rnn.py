#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import h5py
import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

from mdn import MDN, get_mixture_loss_func, get_mixture_mse_accuracy, softmax, \
    get_mixture_sampling_fun
from callbacks import PlotResCallback


def config_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def define_model(N, lstm_layers, vector_size, num_mixtures):
    lstm_params = {'units': N,
                   'activation': 'tanh',
                   'return_sequences': True,
                   'input_shape': (None, vector_size)
                   }
    m = models.Sequential()

    for i in range(lstm_layers):
        m.add(layers.LSTM(**lstm_params))

    m.add(MDN(output_dimension=vector_size, num_mixtures=num_mixtures))
    m.build(lstm_params['input_shape'])
    m.compile(optimizer='rmsprop',
              loss=get_mixture_loss_func(vector_size, num_mixtures))
    return m


def get_dataset(filename):
    with h5py.File(filename, 'r') as f:
        data = f['lines'][:]
    X = data[:, :-1, :].astype(np.float16)
    Y = data[:, 1:, :].astype(np.float16)
    samples, time_steps, vec_size = X.shape
    return X, Y, time_steps


def train(epochs=1):
    X, Y, time_steps = get_dataset('../dataset.h5')
    m = define_model(N, lstm_layers, vector_size, num_mixtures)
    with tf.device('/gpu:0'):
        m.fit(X, Y, epochs=epochs, batch_size=batch_size,
              callbacks=[PlotResCallback(vector_size, num_mixtures, X)]
              )
    m.save('model/hwg_0001.tf')


def load():
    m = models.load_model('model/hwg_0001.tf', custom_objects={
        'mdn_loss_func': get_mixture_loss_func(vector_size, num_mixtures)})
    print(m.summary())
    return m


def sample(m, n, start=None):
    sampling_func = get_mixture_sampling_fun(vector_size, num_mixtures)
    if start is not None:
        # load all, take last
        x = tf.expand_dims(sampling_func(m(start)[0, -1, :]), axis=0)
    else:
        x = np.array([[[12.3124, 25.124124, 0]]])
    res = []

    for i in range(n):
        y = sampling_func(m.predict(x))
        res.append(y[0])
        x = tf.expand_dims(y, axis=1)
    print(res)
    return res


def plot_sample(fig, s, display=True, c='b'):
    x, y = [], []
    cur_x = 0
    cur_y = 0
    # fig.set_size_inches(10, 2)

    for p in s:
        cur_x += p[0]
        cur_y += p[1]
        x.append(cur_x)
        y.append(cur_y)

        # if end-of-stoke
        if p[2] >= 0.5:
            fig.gca().plot(x, y, c)
            x, y = [], []

    fig.gca().plot(x, y, c)
    if display:
        plt.show()


if __name__ == '__main__':
    N = 400
    lstm_layers = 3
    batch_size = 64
    vector_size = 3
    num_mixtures = 20
    config_gpu()
    # train(epochs=30)

    X, _, _= get_dataset('../dataset.h5')
    m = load()
    for i in range(20, 200):
        s = X[i]
        fig = plt.gcf()
        plot_sample(fig, s, display=False, c='r')
        st = tf.expand_dims(s, 0)
        sampler = get_mixture_sampling_fun(vector_size, num_mixtures)
        y_pred = sampler(m(tf.expand_dims(s, 0)))
        plot_sample(fig, y_pred, display=True, c='g')
        fig.savefig(f'media/sample/{i}.jpg')
        plt.show()
