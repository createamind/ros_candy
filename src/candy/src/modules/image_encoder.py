from __future__ import print_function, division, absolute_import
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module

class ImageEncoder(Module):
    def __init__(self, inputs, *args, **kwargs):
        self._inputs = inputs
        super(ImageEncoder, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        x = self._inputs

        if not reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image(self._name, timage[:1])

        with tf.variable_scope('encoder', reuse=reuse) as _:
            # x = B * 320 * 320 * 8 * 3
            x = tf.nn.relu(tf.layers.conv2d(x, 16, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer()))
            # x = B * 160 * 160 * 8 * 32
            x = tf.nn.relu(tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer()))
                
            # x = B * 40 * 40 * 4 * 64
            x = tf.nn.relu(tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer()))

            # x = B * 20 * 20 * 4 * 128
            x = tf.nn.relu(tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='SAME', 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                kernel_initializer=tf.contrib.layers.xavier_initializer()))

            # x = B * 10 * 10 * 2 * 32
            x = tf.reshape(x, [-1, 12800])
            # x = tf.nn.relu(tf.layers.dense(x, 512, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])))
            x = tf.layers.dense(x, 128, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))
            
        return x
