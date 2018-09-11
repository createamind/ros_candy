from __future__ import print_function, division, absolute_import

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module, ModalOps
from modules.image_encoder import ImageEncoder
from modules.image_decoder import ImageDecoder
from modules.utils.utils import mean_square_error


class MultiModal(ModalOps):
    def __init__(self, is_test, args, name, **kwargs):
        self.batch_size = 1 if is_test else args['batch_size']
        self.image_size = args['image_size']
        super(MultiModal, self).__init__(args, name, **kwargs)

    def _build_net(self, is_training):
        self.camera_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, 3), name='camera_x')
        self.eye_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, 3), name='eye_x')
        self.actions = tf.placeholder(tf.float32, shape=(self.batch_size, 2), name='actions')

        self.camera_z = ImageEncoder(self.camera_x, self._args, 'camera_encoder', is_training=is_training, reuse=self._reuse).outputs
        self.eye_z = ImageEncoder(self.eye_x, self._args, 'eye_encoder', is_training=is_training, reuse=self._reuse).outputs
        
        # sample z computation
        with tf.variable_scope('z', reuse=self._reuse):
            self.camera_mean, self.camera_logsigma = tf.split(self.camera_z, 2, 1)
            self.eye_mean, self.eye_logsigma = tf.split(self.eye_z, 2, 1)

            self.mean = tf.concat([self.camera_mean, self.eye_mean], 1)
            logsigma = tf.concat([self.camera_logsigma, self.eye_logsigma], 1)

            sigma = tf.exp(logsigma)
            epsilon = tf.random_normal(tf.shape(sigma))

            self.sample_z = self.mean + sigma * epsilon

            self.camera_sample_z, self.eye_sample_z = tf.split(self.sample_z, 2, 1)

        self.camera_reconstruction = ImageDecoder(self.camera_sample_z, self._args, 'camera_decoder', is_training=is_training, reuse=self._reuse).outputs
        self.eye_reconstruction = ImageDecoder(self.eye_sample_z, self._args, 'eye_decoder', is_training=is_training, reuse=self._reuse).outputs

        with tf.variable_scope('loss', reuse=self._reuse):
            self.KL_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + 2. * logsigma - self.mean**2 - sigma**2, axis=1))
            self.camera_reconstruction_error = mean_square_error(self.camera_x, self.camera_reconstruction)
            self.eye_reconstruction_error = mean_square_error(self.eye_x, self.eye_reconstruction)
            l2_loss = tf.losses.get_regularization_loss()

            MAX_BETA = 1
            beta = tf.get_variable('beta', shape=(), initializer=tf.constant_initializer([0.01]), trainable=False, dtype=tf.float32)
            new_beta = tf.assign(beta, tf.minimum(1.01 * beta, MAX_BETA))
        
            with tf.control_dependencies([new_beta]):
                self.loss = self.camera_reconstruction_error + self.eye_reconstruction_error + beta * self.KL_loss + l2_loss
        
            tf.summary.scalar('beta', beta)
            tf.summary.scalar('KL_loss', self.KL_loss)
            tf.summary.scalar('camera_reconstruction_error', self.camera_reconstruction_error)
            tf.summary.scalar('eye_reconstruction_error', self.eye_reconstruction_error)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total_loss', self.loss)
        
        return self.loss