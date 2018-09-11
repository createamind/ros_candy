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

    def _build_net(self, is_training):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])
        
        def conv_transpose(x, filters, filter_size, strides=1): 
            return tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding='same', 
                                              kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
        def convtrans_bn_relu(x, filters, filter_size, strides=1):
            x = conv_transpose(x, filters, filter_size, strides)
            x = bn_relu(x, is_training)
            return x

        x = self._inputs
        with tf.variable_scope('decoder', reuse=self._reuse) as _:
            x = tf.layers.dense(x, 512, 
                                kernel_initializer=xavier_initializer(), kernel_regularizer=l2_regularizer)
            x = tf.layers.dense(x, 12800, 
                                kernel_initializer=kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = bn_relu(x, is_training)
            
            x = tf.reshape(x, [-1, 5, 5, 512])
            # x = 5, 5, 512
            x = convtrans_bn_relu(x, 256, 3, 2)
            # x = 10, 10, 256
            x = convtrans_bn_relu(x, 128, 3, 2)
            # x = 20, 20, 128
            x = convtrans_bn_relu(x, 64, 5, 2)
            # x = 40, 40, 64
            x = convtrans_bn_relu(x, 32, 5, 2)
            # x = 80, 80, 64
            x = conv_transpose(x, 3, 7, 4)
            x = tf.nn.tanh(x)
            # x = 320, 320, 3
        if not self._reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])
        return x
