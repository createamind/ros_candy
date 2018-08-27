from __future__ import print_function, division, absolute_import

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module


class MultiModal(object):
    def __init__(self, args, is_training, is_test = False):
        batch_size = 1 if is_test else args['batch_size']

        #Placeholders:
        self.camera_left = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
        self.camera_right = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
        self.eye_left = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
        self.eye_right = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
        self.actions = tf.reshape(tf.placeholder(tf.float32, shape=(batch_size, 2, 8)), [batch_size, 16])

        if is_test == False:
            self.camera_left_future = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
            self.camera_right_future = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
            self.eye_left_future = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
            self.eye_right_future = tf.placeholder(tf.float32, shape=(batch_size, 320, 320, 8, 3))
            self.actions_future = tf.reshape(tf.placeholder(tf.float32, shape=(batch_size, 2, 8)), [batch_size, 16])

        #Encoder
        self.encoder_cl = VideoEncoder(args, 'encodercl', self.camera_left, is_training=is_training, reuse=is_test)
        self.encoder_cr = VideoEncoder(args, 'encodercr', self.camera_right, is_training=is_training, reuse=is_test)
        self.encoder_el = VideoEncoder(args, 'encoderel', self.eye_left, is_training=is_training, reuse=is_test)
        self.encoder_er = VideoEncoder(args, 'encoderer', self.eye_right, is_training=is_training, reuse=is_test)
        self.encoder_ac = MLP(10, args, 'encoderac', self.actions, is_training=is_training, reuse=is_test)

        agg = tf.concat([self.encoder_cl.outputs, self.encoder_cr.outputs, self.encoder_el.outputs, self.encoder_er.outputs, self.encoder_ac.outputs], dim=1)
        self.agg = MLP(128, args, 'aggregation', z, is_training=is_training, reuse=is_test)

        # VAE Sampling
        z = self.agg.outputs
        self.mean, self.logsigma = tf.split(z, 2, 1)
        sigma = tf.exp(self.logsigma)
        eps = tf.random_normal(tf.shape(sigma))
        self.sample_z = sigma * eps + self.mean

        #Decoder
        self.de_agg = MLP(410, args, 'de_aggregation', self.sample_z, is_training=is_training, reuse=is_test)

        self.d1 = self.de_agg.outputs[:,0:100]
        self.d2 = self.de_agg.outputs[:,100:200]
        self.d3 = self.de_agg.outputs[:,200:300]
        self.d4 = self.de_agg.outputs[:,300:400]
        self.d5 = self.de_agg.outputs[:,400:410]

        self.decoder_cl = VideoDecoder(args, 'decodercl', self.d1, is_training=is_training, reuse=is_test)
        self.decoder_cr = VideoDecoder(args, 'decodercr', self.d2, is_training=is_training, reuse=is_test)
        self.decoder_el = VideoDecoder(args, 'decoderel', self.d3, is_training=is_training, reuse=is_test)
        self.decoder_er = VideoDecoder(args, 'decoderer', self.d4, is_training=is_training, reuse=is_test)
        self.decoder_ac = MLP(16, args, 'decoderac', self.d5, is_training=is_training, reuse=is_test)

        if is_test == False:
            self.decoder_cl_future = VideoDecoder(args, 'decodercl_future', self.d1, is_training=is_training, reuse=is_test)
            self.decoder_cr_future = VideoDecoder(args, 'decodercr_future', self.d2, is_training=is_training, reuse=is_test)
            self.decoder_el_future = VideoDecoder(args, 'decoderel_future', self.d3, is_training=is_training, reuse=is_test)
            self.decoder_er_future = VideoDecoder(args, 'decoderer_future', self.d4, is_training=is_training, reuse=is_test)
            self.decoder_ac_future = MLP(16, args, 'decoderac_future', self.d5, is_training=is_training, reuse=is_test)

        self.vae_loss = VAELoss(args, 'vae_loss', self.mean, self.logsigma, is_training=is_training, reuse=is_test)

        self.cl_recons = MSELoss(args, 'cl_recons_loss', self.decoder_cl.outputs, self.camera_left, is_training=is_training, reuse=is_test)
        self.cr_recons = MSELoss(args, 'cr_recons_loss', self.decoder_cr.outputs, self.camera_right, is_training=is_training, reuse=is_test)
        self.el_recons = MSELoss(args, 'el_recons_loss', self.decoder_el.outputs, self.eye_left, is_training=is_training, reuse=is_test)
        self.er_recons = MSELoss(args, 'er_recons_loss', self.decoder_er.outputs, self.eye_right, is_training=is_training, reuse=is_test)
        self.ac_recons = MSELoss(args, 'ac_recons_loss', self.decoder_ac.outputs, self.actions, is_training=is_training, reuse=is_test)

        self.loss = 250 * self.vae_loss.outputs + self.cl_recons.outputs + self.cr_recons.outputs + self.el_recons.outputs + self.er_recons.outputs + self.ac_recons.outputs
        
        if is_test == False:
            self.cl_future = MSELoss(args, 'cl_future_loss', self.decoder_cl_future.outputs, self.camera_left_future, is_training=is_training, reuse=is_test)
            self.cr_future = MSELoss(args, 'cr_future_loss', self.decoder_cr_future.outputs, self.camera_right_future, is_training=is_training, reuse=is_test)
            self.el_future = MSELoss(args, 'el_future_loss', self.decoder_el_future.outputs, self.eye_left_future, is_training=is_training, reuse=is_test)
            self.er_future = MSELoss(args, 'er_future_loss', self.decoder_er_future.outputs, self.eye_right_future, is_training=is_training, reuse=is_test)
            self.ac_future = MSELoss(args, 'ac_future_loss', self.decoder_ac_future.outputs, self.actions_future, is_training=is_training, reuse=is_test)

            self.loss += self.cl_future.outputs + self.cr_future.outputs + self.el_future.outputs + self.er_future.outputs + self.ac_future.outputs
