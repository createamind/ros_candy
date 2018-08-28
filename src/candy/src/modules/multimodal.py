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

class MultiModal(Module):
    def __init__(self, is_test, args, name, **kwargs):
        self.batch_size = 1 if is_test else args['batch_size']
        super(MultiModal, self).__init__(args, name, **kwargs)

    def _build_net(self, is_training, reuse):
        #Placeholders:
        self.camera_left = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='camera_left')
        self.camera_right = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='camera_right')
        self.eye_left = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='eye_right')
        self.actions = tf.placeholder(tf.float32, shape=(self.batch_size, 12, 2), name='actions')


        # self.camera_left = tf.Print(self.camera_left, [self.camera_left])
        # if not reuse:
        #     timage = tf.cast((tf.clip_by_value(self.camera_left, -1, 1) + 1) * 127, tf.uint8)
        #     tf.summary.image(self._name + '_first', timage[:1,:,:,0,:])
        #     tf.summary.image(self._name + '_last', timage[:1,:,:,-1,:])

        if reuse == False:
            self.camera_left_future = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='camera_left_future')
            self.camera_right_future = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='camera_right_future')
            self.eye_left_future = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='eye_left_future')
            self.eye_right_future = tf.placeholder(tf.float32, shape=(self.batch_size, 160, 160, 12, 3), name='eye_right_future')
            self.actions_future = tf.placeholder(tf.float32, shape=(self.batch_size, 12, 2), name='actions_future')

        # self.parts = []
        #Encoder
        self.encoder_cl = VideoEncoder(self.camera_left, self._args, 'encodercl', is_training=is_training, reuse=reuse)
        self.encoder_cr = VideoEncoder(self.camera_right, self._args, 'encodercr', is_training=is_training, reuse=reuse)
        self.encoder_el = VideoEncoder(self.eye_left, self._args, 'encoderel', is_training=is_training, reuse=reuse)
        self.encoder_er = VideoEncoder(self.eye_right, self._args, 'encoderer', is_training=is_training, reuse=reuse)
        self.encoder_ac = MLP(tf.reshape(self.actions, [self.batch_size, 24]), [10], self._args, 'encoderac', is_training=is_training, reuse=reuse)
        # self.parts += [self.encoder_cl, self.encoder_cr, self.encoder_el, self.encoder_er, self.encoder_ac]

        self.agg = tf.concat([self.encoder_cl.outputs, self.encoder_cr.outputs, self.encoder_el.outputs, self.encoder_er.outputs, self.encoder_ac.outputs], 1)
        self.agg = tf.layers.dropout(self.agg, rate=0.25, training=is_training)
        self.agg = MLP(self.agg, [128], self._args, 'aggregation',is_training=is_training, reuse=reuse)
        # self.parts += [self.agg]

        # VAE Sampling
        z = self.agg.outputs
        self.mean, self.logsigma = tf.split(z, 2, 1)
        sigma = tf.exp(self.logsigma)
        eps = tf.random_normal(tf.shape(sigma))
        self.sample_z = sigma * eps + self.mean

        #Decoder
        self.de_mlp = MLP(self.sample_z, [522], self._args, 'de_aggregation', is_training=is_training, reuse=reuse)
        self.de_agg = tf.layers.dropout(self.de_mlp.outputs, rate=0.25, training=is_training)
        # self.parts += [self.de_agg]

        self.d1 = self.de_agg[:,0:128]
        self.d2 = self.de_agg[:,128:256]
        self.d3 = self.de_agg[:,256:384]
        self.d4 = self.de_agg[:,384:512]
        self.d5 = self.de_agg[:,512:522]

        self.decoder_cl = VideoDecoder(self.d1, self._args, 'decodercl', is_training=is_training, reuse=reuse)
        self.decoder_cr = VideoDecoder(self.d2, self._args, 'decodercr', is_training=is_training, reuse=reuse)
        self.decoder_el = VideoDecoder(self.d3, self._args, 'decoderel', is_training=is_training, reuse=reuse)
        self.decoder_er = VideoDecoder(self.d4, self._args, 'decoderer', is_training=is_training, reuse=reuse)
        self.decoder_ac = MLP(self.d5, [24], self._args, 'decoderac', is_training=is_training, reuse=reuse)
        # self.parts += [self.decoder_cl, self.decoder_cr, self.decoder_el, self.decoder_er, self.decoder_ac]


        if reuse == False:
            self.decoder_cl_future = VideoDecoder(self.d1, self._args, 'decodercl_future', is_training=is_training, reuse=reuse)
            self.decoder_cr_future = VideoDecoder(self.d2, self._args, 'decodercr_future', is_training=is_training, reuse=reuse)
            self.decoder_el_future = VideoDecoder(self.d3, self._args, 'decoderel_future', is_training=is_training, reuse=reuse)
            self.decoder_er_future = VideoDecoder(self.d4, self._args, 'decoderer_future', is_training=is_training, reuse=reuse)
            self.decoder_ac_future = MLP(self.d5, [24], self._args, 'decoderac_future', is_training=is_training, reuse=reuse)
            # self.parts += [self.decoder_cl_future, self.decoder_cr_future, self.decoder_el_future, self.decoder_er_future, self.decoder_ac_future]

        self.vae_loss = VAELoss(self.mean, self.logsigma, self._args, 'vae_loss', is_training=is_training, reuse=reuse)

        self.cl_recons = MSELoss(self.decoder_cl.outputs, self.camera_left, self._args, 'cl_recons_loss', is_training=is_training, reuse=reuse)
        self.cr_recons = MSELoss(self.decoder_cr.outputs, self.camera_right, self._args, 'cr_recons_loss', is_training=is_training, reuse=reuse)
        self.el_recons = MSELoss(self.decoder_el.outputs, self.eye_left, self._args, 'el_recons_loss', is_training=is_training, reuse=reuse)
        self.er_recons = MSELoss(self.decoder_er.outputs, self.eye_right, self._args, 'er_recons_loss', is_training=is_training, reuse=reuse)
        self.ac_recons = MSELoss(self.decoder_ac.outputs, tf.reshape(self.actions, [self.batch_size, 24]), self._args, 'ac_recons_loss', is_training=is_training, reuse=reuse)

        self.loss = 250 * self.vae_loss.outputs + self.cl_recons.outputs + self.cr_recons.outputs + self.el_recons.outputs + self.er_recons.outputs + self.ac_recons.outputs
        
        if reuse == False:
            self.cl_future = MSELoss(self.decoder_cl_future.outputs, self.camera_left_future, self._args, 'cl_future_loss', is_training=is_training, reuse=reuse)
            self.cr_future = MSELoss(self.decoder_cr_future.outputs, self.camera_right_future, self._args, 'cr_future_loss', is_training=is_training, reuse=reuse)
            self.el_future = MSELoss(self.decoder_el_future.outputs, self.eye_left_future, self._args, 'el_future_loss', is_training=is_training, reuse=reuse)
            self.er_future = MSELoss(self.decoder_er_future.outputs, self.eye_right_future, self._args, 'er_future_loss', is_training=is_training, reuse=reuse)
            self.ac_future = MSELoss(self.decoder_ac_future.outputs, tf.reshape(self.actions_future, [self.batch_size, 24]), self._args, 'ac_future_loss', is_training=is_training, reuse=reuse)

            self.loss += self.cl_future.outputs + self.cr_future.outputs + self.el_future.outputs + self.er_future.outputs + self.ac_future.outputs

        return self.loss
    #     #Variable saver
    #     collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
    #     if len(collection) > 0:
    #         self._saver = tf.train.Saver(collection)
    #     else:
    #         self._saver = None
            
    # def optimize(self, loss):

    #     self.ops = []
    #     for part in self.parts:
    #         self.ops.append(part.optimize(loss))
    #     self.ops = tf.group(self.ops)

    #     return self.ops
 
    # def variable_restore(self, sess):
    #     for part in self.parts:
    #         part.variable_restore(sess)

    
    # def save(self, sess):
    #     for part in self.parts:
    #         part.save(sess)