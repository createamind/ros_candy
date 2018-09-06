from __future__ import print_function, division, absolute_import
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module
from modules.utils.utils import kaiming_initializer, xavier_initializer, bn_relu

class ImageEncoder(Module):
    def __init__(self, inputs, *args, **kwargs):
        self._inputs = inputs
        super(ImageEncoder, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])

        x = self._inputs

        if not reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])

        with tf.variable_scope('encoder', reuse=reuse) as _:
            # x = 320, 320, 3
            x = tf.layers.conv2d(x, 32, (7, 7), strides=(4, 4), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            # x = 80, 80, 32
            x = tf.layers.conv2d(x, 64, (5, 5), strides=(4, 4), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer,)
            x = bn_relu(x, is_training)

            # x = 20, 20, 64
            x = tf.layers.conv2d(x, 128, (5, 5), strides=(2, 2), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)

            # x = 10, 10, 128
            x = tf.layers.conv2d(x, 256, (3, 3), strides=(2, 2), padding='same', 
                             kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)

            # x = 5, 5, 256
            x = tf.reshape(x, [-1, 6400])

            x = tf.layers.dense(x, 512, 
                                kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            x = tf.layers.dense(x, 128, kernel_initializer=xavier_initializer(), kernel_regularizer=l2_regularizer)

        return x
