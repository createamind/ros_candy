from __future__ import print_function, division, absolute_import
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module
from modules.utils.utils import kaiming_initializer, xavier_initializer, bn_relu

class ImageDecoder(Module):
    def __init__(self, inputs, *args, **kwargs):
        self._inputs = inputs
        super(ImageDecoder, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])

        x = self._inputs
        with tf.variable_scope('decoder', reuse=reuse) as _:
            x = tf.layers.dense(x, 512, 
                                kernel_initializer=xavier_initializer(), kernel_regularizer=l2_regularizer)
            x = tf.layers.dense(x, 6400, 
                                kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            
            x = tf.reshape(x, [-1, 5, 5, 256])
            # x = 5, 5, 256
            x = tf.layers.conv2d(x, 128, (3, 3), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            x = tf.image.resize_nearest_neighbor(x, (10, 10))

            # x = 10, 10, 128
            x = tf.layers.conv2d(x, 64, (5, 5), strides=(2, 2), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            x = tf.image.resize_nearest_neighbor(x, (20, 20))

            # x = 20, 20, 64
            x = tf.layers.conv2d(x, 32, (5, 5), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            x = tf.image.resize_nearest_neighbor(x, (80, 80))

            # x = 80, 80, 32
            x = tf.layers.conv2d(x, 3, (7, 7), padding='same', 
                                 kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = tf.image.resize_nearest_neighbor(x, (320, 320))
            x = tf.nn.tanh(x)
            # x = 320, 320, 8, 3
        if not reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])
        return x
