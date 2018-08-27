from __future__ import print_function, division, absolute_import

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module
from modules.video_encoder import VideoEncoder
from modules.video_decoder import VideoDecoder
from modules.losses import MSELoss, VAELoss
from modules.mlp import MLP

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

        self.parts = []
        #Encoder
        self.encoder_cl = VideoEncoder(self.camera_left, args, 'encodercl', is_training=is_training, reuse=is_test)
        self.encoder_cr = VideoEncoder(self.camera_right, args, 'encodercr', is_training=is_training, reuse=is_test)
        self.encoder_el = VideoEncoder(self.eye_left, args, 'encoderel', is_training=is_training, reuse=is_test)
        self.encoder_er = VideoEncoder(self.eye_right, args, 'encoderer', is_training=is_training, reuse=is_test)
        self.encoder_ac = MLP(self.actions, [10], args, 'encoderac', is_training=is_training, reuse=is_test)
        self.parts += [self.encoder_cl, self.encoder_cr, self.encoder_el, self.encoder_er, self.encoder_ac]

        agg = tf.concat([self.encoder_cl.outputs, self.encoder_cr.outputs, self.encoder_el.outputs, self.encoder_er.outputs, self.encoder_ac.outputs], 1)
        self.agg = MLP(agg, [128], args, 'aggregation',is_training=is_training, reuse=is_test)
        self.parts += [self.agg]

        # VAE Sampling
        z = self.agg.outputs
        self.mean, self.logsigma = tf.split(z, 2, 1)
        sigma = tf.exp(self.logsigma)
        eps = tf.random_normal(tf.shape(sigma))
        self.sample_z = sigma * eps + self.mean

        #Decoder
        self.de_agg = MLP(self.sample_z, [522], args, 'de_aggregation', is_training=is_training, reuse=is_test)
        self.parts += [self.de_agg]

        self.d1 = self.de_agg.outputs[:,0:128]
        self.d2 = self.de_agg.outputs[:,128:256]
        self.d3 = self.de_agg.outputs[:,256:384]
        self.d4 = self.de_agg.outputs[:,384:512]
        self.d5 = self.de_agg.outputs[:,512:522]

        self.decoder_cl = VideoDecoder(self.d1, args, 'decodercl', is_training=is_training, reuse=is_test)
        self.decoder_cr = VideoDecoder(self.d2, args, 'decodercr', is_training=is_training, reuse=is_test)
        self.decoder_el = VideoDecoder(self.d3, args, 'decoderel', is_training=is_training, reuse=is_test)
        self.decoder_er = VideoDecoder(self.d4, args, 'decoderer', is_training=is_training, reuse=is_test)
        self.decoder_ac = MLP(self.d5, [16], args, 'decoderac', is_training=is_training, reuse=is_test)
        self.parts += [self.decoder_cl, self.decoder_cr, self.decoder_el, self.decoder_er, self.decoder_ac]


        if is_test == False:
            self.decoder_cl_future = VideoDecoder(self.d1, args, 'decodercl_future', is_training=is_training, reuse=is_test)
            self.decoder_cr_future = VideoDecoder(self.d2, args, 'decodercr_future', is_training=is_training, reuse=is_test)
            self.decoder_el_future = VideoDecoder(self.d3, args, 'decoderel_future', is_training=is_training, reuse=is_test)
            self.decoder_er_future = VideoDecoder(self.d4, args, 'decoderer_future', is_training=is_training, reuse=is_test)
            self.decoder_ac_future = MLP(self.d5, [16], args, 'decoderac_future', is_training=is_training, reuse=is_test)
            self.parts += [self.decoder_cl_future, self.decoder_cr_future, self.decoder_el_future, self.decoder_er_future, self.decoder_ac_future]

        self.vae_loss = VAELoss(self.mean, self.logsigma, args, 'vae_loss', is_training=is_training, reuse=is_test)

        self.cl_recons = MSELoss(self.decoder_cl.outputs, self.camera_left, args, 'cl_recons_loss', is_training=is_training, reuse=is_test)
        self.cr_recons = MSELoss(self.decoder_cr.outputs, self.camera_right, args, 'cr_recons_loss', is_training=is_training, reuse=is_test)
        self.el_recons = MSELoss(self.decoder_el.outputs, self.eye_left, args, 'el_recons_loss', is_training=is_training, reuse=is_test)
        self.er_recons = MSELoss(self.decoder_er.outputs, self.eye_right, args, 'er_recons_loss', is_training=is_training, reuse=is_test)
        self.ac_recons = MSELoss(self.decoder_ac.outputs, self.actions, args, 'ac_recons_loss', is_training=is_training, reuse=is_test)

        self.loss = 250 * self.vae_loss.outputs + self.cl_recons.outputs + self.cr_recons.outputs + self.el_recons.outputs + self.er_recons.outputs + self.ac_recons.outputs
        
        if is_test == False:
            self.cl_future = MSELoss(self.decoder_cl_future.outputs, self.camera_left_future, args, 'cl_future_loss', is_training=is_training, reuse=is_test)
            self.cr_future = MSELoss(self.decoder_cr_future.outputs, self.camera_right_future, args, 'cr_future_loss', is_training=is_training, reuse=is_test)
            self.el_future = MSELoss(self.decoder_el_future.outputs, self.eye_left_future, args, 'el_future_loss', is_training=is_training, reuse=is_test)
            self.er_future = MSELoss(self.decoder_er_future.outputs, self.eye_right_future, args, 'er_future_loss', is_training=is_training, reuse=is_test)
            self.ac_future = MSELoss(self.decoder_ac_future.outputs, self.actions_future, args, 'ac_future_loss', is_training=is_training, reuse=is_test)

            self.loss += self.cl_future.outputs + self.cr_future.outputs + self.el_future.outputs + self.er_future.outputs + self.ac_future.outputs

    def optimize(self, loss):

        self.ops = []
        for part in self.parts:
            self.ops.append(part.optimize(loss))
        self.ops = tf.group(self.ops)
 
    def variable_restore(self, sess):
        for part in self.parts:
            part.variable_restore(sess)