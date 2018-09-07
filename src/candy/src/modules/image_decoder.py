from __future__ import print_function, division, absolute_import
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module

class ImageDecoder(Module):
    def __init__(self, inputs, *args, **kwargs):
        self._inputs = inputs
        super(ImageDecoder, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        x = self._inputs
        with tf.variable_scope('decoder', reuse=reuse) as _:
            # x = tf.nn.leaky_relu(tf.layers.dense(x, 512, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])))
            x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(tf.layers.dense(x, 51200, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))))

            x = tf.reshape(x, [-1, 10, 10, 512])

            x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(tf.layers.conv2d_transpose(x, 256, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer())))

            x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer())))

            x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(tf.layers.conv2d_transpose(x, 64, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer())))

            x = tf.nn.leaky_relu(tf.contrib.layers.instance_norm(tf.layers.conv2d_transpose(x, 16, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer())))

            x = tf.nn.tanh(tf.layers.conv2d_transpose(x, 3, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer()))

        if not reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])
        return x
