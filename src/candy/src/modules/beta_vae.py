import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import modules.utils.utils as utils
from modules.module import Module

class BetaVAE(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        self.z_size = args['z_size']
        self.image_size = args['image_size']
        super(BetaVAE, self).__init__(name, args, reuse)

    def generate(self, sess, num_images=1):
        with tf.variable_scope(self._name, reuse=True):
            sample_z = np.random.normal(size=(num_images, self.z_size))
            return sess.run(self.x_mu, feed_dict={self.sample_z: sample_z})

    """" Implementation """
    def _build_graph(self):
        with tf.variable_scope('placeholder', reuse=self.reuse):
            self.inputs = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3), name='inputs')
            self.is_training = tf.placeholder(tf.bool, (None), name='is_training')
                
        self.z_mu, self.z_logsigma = self._encode()
        self.sample_z = self._sample_norm(self.z_mu, self.z_logsigma, 'sample_z')
        self.x_mu = self._decode(self.sample_z, reuse=self.reuse)
        self.loss = self._loss(self.z_mu, self.z_logsigma, self.x_mu, self.inputs)

        # add image summaries at training time
        with tf.name_scope('image'):
                # record an original image
                timage = tf.cast((tf.clip_by_value(self.inputs, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('original_image', timage[:1])
                # record a generated image
                timage = tf.cast((tf.clip_by_value(self.x_mu, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('generated_image', timage[:1])

    def _encode(self):                                 
        x = self.inputs
        
        # encoder net
        with tf.variable_scope('encoder', reuse=self.reuse):
            x = self._conv_bn_relu(x, 16, 4, 2)                  # x = 160, 160, 16
            x = self._conv_bn_relu(x, 32, 4, 2)                  # x = 80, 80, 32
            x = self._conv_bn_relu(x, 64, 4, 2)                  # x = 40, 40, 64
            x = self._conv_bn_relu(x, 128, 4, 2)                 # x = 20, 20, 128
            x = self._conv_bn_relu(x, 256, 4, 2)                 # x = 10, 10, 256
            x = self._conv_bn_relu(x, 512, 4, 2)                 # x = 5, 5, 512
            self.dim_feature_map = x
            """ Version without dense layer """
            x = self._conv(x, 2 * self.z_size, 5, padding='valid', kernel_initializer=utils.xavier_initializer())  
            # x = 1, 1, 2 * z_size

            x = tf.reshape(x, [-1, 2 * self.z_size])
            
            mu, logsigma = tf.split(x, 2, -1)

            # record some weights
        if not self.reuse:
            with tf.variable_scope('encoder', reuse=True):
                w = tf.get_variable('conv2d/kernel')
                tf.summary.histogram('conv0_weights', w)

        return mu, logsigma

    def _sample_norm(self, mu, logsigma, scope, reuse=None):
        with tf.variable_scope(scope, reuse=self.reuse if reuse is None else reuse):
            sigma = tf.exp(logsigma)
            epsilon = tf.random_normal(tf.shape(mu))

            sample = mu + sigma * epsilon
            tf.summary.histogram(scope, sample)

        return sample

    def _decode(self, sample_z, reuse):
        x = sample_z

        # decoder net
        with tf.variable_scope('decoder', reuse=reuse):
            """ Version without dense layer """
            x = tf.reshape(x, [-1, 1, 1, self.z_size])                  # x = 1, 1, z_size

            x = self._convtrans_bn_relu(x, 512, 5, 1, padding='valid')   # x = 5, 5, 512

            x = self._convtrans_bn_relu(x, 256, 4, 2)                    # x = 10, 10, 256
            x = self._convtrans_bn_relu(x, 128, 4, 2)                    # x = 20, 20, 128
            x = self._convtrans_bn_relu(x, 64, 4, 2)                     # x = 40, 40, 64
            x = self._convtrans_bn_relu(x, 32, 4, 2)                     # x = 80, 80, 32
            x = self._convtrans_bn_relu(x, 16, 4, 2)                     # x = 160, 160, 16
            x = self._convtrans(x, 3, 4, 2, kernel_initializer=utils.xavier_initializer())
            # x = 320, 320, 3
            x = tf.tanh(x)

            x_mu = x
            
        if not self.reuse:
            with tf.variable_scope('decoder', reuse=True):
                # record some weights
                w = tf.get_variable('conv2d_transpose_5/kernel')
                tf.summary.histogram('convtrans5_weights', w)

        return x_mu

    def _loss(self, mu, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                KL_loss = utils.kl_loss(mu, logsigma)
                MAX_BETA = self._args[self._name]['beta']
                beta = tf.get_variable('beta', shape=(), initializer=tf.constant_initializer([1e-6]), trainable=False, dtype=tf.float32)
                new_beta = tf.assign(beta, tf.minimum(1.0005 * beta, MAX_BETA))
                beta_KL = beta * KL_loss

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            with tf.variable_scope('l2_regulization', reuse=self.reuse):
                l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.control_dependencies([new_beta]):
                loss = reconstruction_loss + beta_KL + l2_loss

            tf.summary.scalar('reconstruction_error', reconstruction_loss)
            tf.summary.scalar('beta', beta)
            tf.summary.scalar('KL_loss', KL_loss)
            tf.summary.scalar('beta_KL', beta_KL)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total_loss', loss)

        return loss

    def _optimize(self, loss):
        # params for optimizer
        init_learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999

        with tf.variable_scope('optimizer', reuse=self.reuse):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.get_variable('global_step', shape=(), initializer=0, trainable=False)
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 1000, 0.95, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            tf.summary.scalar('learning_rate', learning_rate)

        with tf.control_dependencies(update_ops):
            opt_op = self._optimizer.minimize(loss, global_step=global_step)

        return opt_op
