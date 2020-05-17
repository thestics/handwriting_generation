#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.mdn import get_mixture_sampling_fun


class PlotResCallback(tf.keras.callbacks.Callback):

    SAMPLE_SIZE = 500
    # SAMPLING_START_POINT = np.array([[[0.1, 0.1, 0]]])

    def __init__(self, vec_size, num_mixes, X):
        self.vec_size = vec_size
        self.num_mixes = num_mixes
        self.X = X[0]
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        fig = plt.gcf()
        self.plot_sample(fig, self.X, epoch)
        s = self.sample()
        self.plot_sample(fig, s, epoch, c='r', display=True)
        fig.savefig(f'media/epoch-{epoch}.jpg')
        plt.show()

    def sample(self):
        sampling_func = get_mixture_sampling_fun(self.vec_size, self.num_mixes)
        res = self.model(tf.expand_dims(self.X, axis=0))
        x = sampling_func(res)
        return x

    def plot_sample(self, fig, s, epoch, display=True, c='b'):
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
            fig.savefig(f'media/epoch-{epoch}.jpg')