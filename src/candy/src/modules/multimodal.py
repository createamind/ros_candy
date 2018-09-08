from __future__ import print_function, division, absolute_import

import sys
import tensorflow as tf
import numpy as np
import math
import os
from modules.module import Module
from modules.image_encoder import ImageEncoder
from modules.image_decoder import ImageDecoder
from modules.losses import MSELoss, VAELoss
from modules.mlp import MLP

class MultiModal(Module):
    def __init__(self, is_test, args, name, **kwargs):
        self.batch_size = 1 if is_test else args['batch_size']
        super(MultiModal, self).__init__(args, name, **kwargs)

    def _build_net(self, is_training, reuse):
        #Placeholders:
        self.camera_left = tf.placeholder(tf.float32, shape=(self.batch_size, 320, 320, 3), name='camera_left')
        self.camera_right = tf.placeholder(tf.float32, shape=(self.batch_size, 320, 320, 3), name='camera_right')
        self.eye_left = tf.placeholder(tf.float32, shape=(self.batch_size, 320, 320, 3), name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, shape=(self.batch_size, 320, 320, 3), name='eye_right')
        self.actions = tf.placeholder(tf.float32, shape=(self.batch_size, 2), name='actions')

        # self.parts = []
        #Encoder
        self.encoder_cl = ImageEncoder(self.camera_left, self._args, 'encodercl', is_training=is_training, reuse=reuse)
        self.encoder_cr = ImageEncoder(self.camera_right, self._args, 'encodercr', is_training=is_training, reuse=reuse)
        self.encoder_el = ImageEncoder(self.eye_left, self._args, 'encoderel', is_training=is_training, reuse=reuse)
        self.encoder_er = ImageEncoder(self.eye_right, self._args, 'encoderer', is_training=is_training, reuse=reuse)
        # self.encoder_ac = MLP(self.actions, [10], self._args, 'encoderac', is_training=is_training, reuse=reuse)
        # self.parts += [self.encoder_cl, self.encoder_cr, self.encoder_el, self.encoder_er, self.encoder_ac]

        # self.agg = tf.concat([self.encoder_cl.outputs, self.encoder_cr.outputs, self.encoder_el.outputs, self.encoder_er.outputs, self.encoder_ac.outputs], 1)
        self.e1 = tf.layers.dropout(self.encoder_cl.outputs, rate=0.25, training=is_training)
        self.e2 = tf.layers.dropout(self.encoder_cr.outputs, rate=0.25, training=is_training)
        self.e3 = tf.layers.dropout(self.encoder_el.outputs, rate=0.25, training=is_training)
        self.e4 = tf.layers.dropout(self.encoder_er.outputs, rate=0.25, training=is_training)
        
        self.mean1, self.logsigma1 = tf.split(self.e1, 2, 1)
        self.mean2, self.logsigma2 = tf.split(self.e2, 2, 1)
        self.mean3, self.logsigma3 = tf.split(self.e3, 2, 1)
        self.mean4, self.logsigma4 = tf.split(self.e4, 2, 1)

        # self.e1 = tf.layers.dropout(self.agg, rate=0.25, training=is_training)
        # self.agg = MLP(self.agg, [128], self._args, 'aggregation',is_training=is_training, reuse=reuse)
        # self.parts += [self.agg]

        # VAE Sampling
        # z = self.agg.outputs


        self.mean = tf.concat([self.mean1, self.mean2, self.mean3, self.mean4], 1)
        self.z = tf.concat([self.mean1[:,:4], self.mean2[:,:4], self.mean3[:,:4], self.mean4[:,:4]], 1)
        self.logsigma = tf.concat([self.logsigma1, self.logsigma2, self.logsigma3, self.logsigma4], 1)

        # self.mean, self.logsigma = tf.split(z, 2, 1)
        sigma = tf.exp(self.logsigma)
        eps = tf.random_normal(tf.shape(sigma))
        self.sample_z = sigma * eps + self.mean

        self.d1, self.d2, self.d3, self.d4 = tf.split(self.sample_z, 4, 1)
        #Decoder
        # self.de_mlp = MLP(self.sample_z, [522], self._args, 'de_aggregation', is_training=is_training, reuse=reuse)
        # self.de_agg = tf.layers.dropout(self.de_mlp.outputs, rate=0.25, training=is_training)
        # self.parts += [self.de_agg]

        # self.d1 = self.de_agg[:,0:128]
        # self.d2 = self.de_agg[:,128:256]
        # self.d3 = self.de_agg[:,256:384]
        # self.d4 = self.de_agg[:,384:512]
        # self.d5 = self.de_agg[:,512:522]

        self.decoder_cl = ImageDecoder(self.d1, self._args, 'decodercl', is_training=is_training, reuse=reuse)
        self.decoder_cr = ImageDecoder(self.d2, self._args, 'decodercr', is_training=is_training, reuse=reuse)
        self.decoder_el = ImageDecoder(self.d3, self._args, 'decoderel', is_training=is_training, reuse=reuse)
        self.decoder_er = ImageDecoder(self.d4, self._args, 'decoderer', is_training=is_training, reuse=reuse)
        # self.decoder_ac = MLP(self.d5, [2], self._args, 'decoderac', is_training=is_training, reuse=reuse)
        # self.parts += [self.decoder_cl, self.decoder_cr, self.decoder_el, self.decoder_er, self.decoder_ac]

        self.vae_loss = VAELoss(self.mean, self.logsigma, self._args, 'vae_loss', is_training=is_training, reuse=reuse)

        self.cl_recons = MSELoss(self.decoder_cl.outputs, self.camera_left, self._args, 'cl_recons_loss', is_training=is_training, reuse=reuse)
        self.cr_recons = MSELoss(self.decoder_cr.outputs, self.camera_right, self._args, 'cr_recons_loss', is_training=is_training, reuse=reuse)
        self.el_recons = MSELoss(self.decoder_el.outputs, self.eye_left, self._args, 'el_recons_loss', is_training=is_training, reuse=reuse)
        self.er_recons = MSELoss(self.decoder_er.outputs, self.eye_right, self._args, 'er_recons_loss', is_training=is_training, reuse=reuse)
        # self.ac_recons = MSELoss(self.decoder_ac.outputs, self.actions, self._args, 'ac_recons_loss', is_training=is_training, reuse=reuse)

        beta = 1
        self.loss = beta * self.vae_loss.outputs + self.cl_recons.outputs + self.cr_recons.outputs + self.el_recons.outputs + self.er_recons.outputs
        
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