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
        
        def conv(x, filters, filter_size, strides=1): 
            return tf.layers.conv2d(x, filters, filter_size, strides=strides, padding='same', 
                                    kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
        def conv_bn_relu(x, filters, filter_size, strides=1):
            x = conv(x, filters, filter_size, strides)
            x = bn_relu(x, is_training)
            return x

        x = self._inputs
        
        if not reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])

        with tf.variable_scope('encoder', reuse=reuse) as _:
            # x = 320, 320, 3
            x = conv_bn_relu(x, 64, 7, 4)

            # x = 80, 80, 64
            x = conv_bn_relu(x, 128, 5, 4)

            # x = 20, 20, 128
            x = conv_bn_relu(x, 256, 3, 2)

            # x = 10, 10, 256
            x = conv_bn_relu(x, 512, 3, 2)
            
            # x = 5, 5, 256
            x = tf.reshape(x, [-1, 12800])

            x = tf.layers.dense(x, 512, 
                                kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            x = tf.layers.dense(x, 128, kernel_initializer=xavier_initializer(), kernel_regularizer=l2_regularizer)

        return x
