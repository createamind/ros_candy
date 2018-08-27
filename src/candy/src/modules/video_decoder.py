from __future__ import print_function, division, absolute_import
from scipy.misc import imsave

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module

class VAE3D(Module):
    def __init__(self, *args, **kwargs):
        super(VAE3D, self).__init__(*args, **kwargs)

    def encoder(self, x, is_training, reuse):
        """Define q(z|x) network"""

        with tf.variable_scope(self._name, reuse=reuse) as _:
            with tf.variable_scope('encoder', reuse=reuse) as _2:
                # x = B * 320 * 320 * 8 * 3
                x = tf.nn.relu(tf.layers.conv3d(x, 8, [4, 4, 4], strides=(2, 2, 1), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()))
                # x = B * 160 * 160 * 8 * 32
                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d(x, 8, [4, 4, 4], strides=(2, 2, 2), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))
                
                # x = B * 80 * 80 * 4 * 32
                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d(x, 16, [4, 4, 4], strides=(2, 2, 1), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                # x = B * 40 * 40 * 4 * 64
                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d(x, 16, [4, 4, 4], strides=(2, 2, 1), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                # x = B * 20 * 20 * 4 * 64
                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d(x, 32, [4, 4, 4], strides=(2, 2, 2), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                # x = B * 10 * 10 * 2 * 32
                x = tf.reshape(x, [-1, 6400])
                # x = tf.nn.relu(tf.layers.dense(x, 512, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])))
                z = tf.layers.dense(x, 128, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))
                
                mean, logsigma = tf.split(z, 2, 1)

        return mean, logsigma


    def decoder(self, mean, logsigma, is_training, reuse):
        sigma = tf.exp(logsigma)
        eps = tf.random_normal(tf.shape(sigma))
        x = sigma * eps + mean

        with tf.variable_scope(self._name, reuse=reuse) as _:
            with tf.variable_scope('decoder', reuse=reuse) as _2:

                # x = tf.nn.relu(tf.layers.dense(x, 512, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])))
                x = tf.nn.relu(tf.layers.dense(x, 6400, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])))

                x = tf.reshape(x, [-1, 10, 10, 2, 32])

                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d_transpose(x, 16, [4, 4, 4], strides=(2, 2, 2), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d_transpose(x, 16, [4, 4, 4], strides=(2, 2, 1), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d_transpose(x, 8, [4, 4, 4], strides=(2, 2, 1), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                x = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv3d_transpose(x, 8, [4, 4, 4], strides=(2, 2, 2), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

                x = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv3d_transpose(x, 3, [4, 4, 4], strides=(2, 2, 2), padding='SAME', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']),
                    kernel_initializer=tf.contrib.layers.xavier_initializer()), training=is_training))

        return tf.split(x, 2, 3)

    def _build_net(self, inputs, is_training, reuse):

        timage = tf.cast((tf.clip_by_value(inputs, -1, 1) + 1) * 127, tf.uint8)
        print(timage[0,:,:,0,:])
        for i in range(1):
            tf.summary.image("image_real", timage[:1,:,:,i,:])
        # tf.summary.image("image_real", timage[:,:,:,4:])

        mean, logsigma = self.encoder(inputs, is_training, reuse)
        recon_x, future_x = self.decoder(mean, logsigma, is_training, reuse)

        timage = tf.cast((tf.clip_by_value(recon_x, -1, 1) + 1) * 127, tf.uint8)
        for i in range(1):
            tf.summary.image("image_recon", timage[:1,:,:,i,:])

        timage = tf.cast((tf.clip_by_value(future_x, -1, 1) + 1) * 127, tf.uint8)
        for i in range(1):
            tf.summary.image("image_prediction", timage[:1,:,:,i,:])

        # tf.summary.image("image_recon", timage[:,:,:,:3])
        # tf.summary.image("image_recon", timage[:,:,:,4:])

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name))

        return recon_x, future_x, mean, logsigma
