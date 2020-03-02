#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return keras.backend.elu(x) + 1 + keras.backend.epsilon()


def biased_softmax(bias=0):
    def activation(x):
        return keras.activations.softmax(x * (1. + bias))
    return activation


def biased_exp(bias=0):
    def activation(x):
        return tf.exp(x - bias)
    return activation



class MDN(layers.Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, bias=0, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

        with tf.name_scope('MDN'):
            # end of stroke probability
            self.mdn_e = layers.Dense(1, name='mdn_e', activation='sigmoid')
            # mixing values, logits
            self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi',
                                       activation=biased_softmax(bias))
            # means
            self.mdn_mu1 = layers.Dense(self.num_mix, name='mdn_mu1')
            self.mdn_mu2 = layers.Dense(self.num_mix, name='mdn_mu2')
            # std`s
            self.mdn_std1 = layers.Dense(self.num_mix, name='mdn_std1',
                                         activation=biased_exp(bias))
            self.mdn_std2 = layers.Dense(self.num_mix, name='mdn_std2',
                                         activation=biased_exp(bias))
            # correlation
            self.mdn_rho = layers.Dense(self.num_mix, name='mdn_rho',
                                        activation='tanh')
            self.layers = [self.mdn_e, self.mdn_pi, self.mdn_mu1, self.mdn_mu2,
                           self.mdn_std1, self.mdn_std2, self.mdn_rho]
        super(MDN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.name_scope('layers'):
            for layer in self.layers:
                layer.build(input_shape)
        super(MDN, self).build(input_shape)


    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([l(x) for l in self.layers],
                                         name='mdn_outputs')
        return mdn_out



def get_custom_mixture_loss_func(num_mixes, eps=1e-8):
    """
    Construct a loss functions for the MDN layer parametrised
    by number of mixtures.
    """

    def mdn_loss_func(y_true, y_pred):
        # Split the inputs into parameters, 1 for end-of-stroke, `num_mixes`
        # for other
        # y_true = tf.reshape(tensor=y_true, shape=y_pred.shape)
        origin_shape = tf.shape(y_true)
        new_shape = [origin_shape[0], origin_shape[1], 3]
        y_true = tf.reshape(y_true, new_shape)
        e, pi, mu1, mu2, std1, std2, rho = tf.split(
            y_pred, num_or_size_splits=[1, ] + [num_mixes for _ in range(6)],
            name='mdn_coef_split', axis=-1
        )
        cat = tfd.Categorical(probs=pi)
        mu1_comps = [tf.expand_dims(c, -1) for c in tf.unstack(mu1, axis=-1)]
        mu2_comps = [tf.expand_dims(c, -1) for c in tf.unstack(mu2, axis=-1)]
        std1_comps = [tf.expand_dims(c, -1) for c in tf.unstack(std1, axis=-1)]
        std2_comps = [tf.expand_dims(c, -1) for c in tf.unstack(std2, axis=-1)]
        components = [tfd.MultivariateNormalDiag(loc=[mu1_i, mu2_i], scale_diag=[std1_i, std2_i])
                        for mu1_i, mu2_i, std1_i, std2_i
                        in zip(mu1_comps, mu2_comps,
                               std1_comps, std2_comps)
                     ]
        mix = tfd.Mixture(cat=cat, components=components)
        xs, ys, es = tf.unstack(y_true, axis=-1)
        X = tf.stack((xs, ys), axis=-1)
        stroke = tfd.Bernoulli(probs=[e,])
        loss1 = tf.negative(mix.log_prob(X))
        loss2 = tf.negative(stroke.log_prob(es))
        loss = tf.add(loss1, loss2)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func
