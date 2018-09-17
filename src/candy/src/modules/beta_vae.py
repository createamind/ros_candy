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
        with tf.name_scope(self._name):
            sample_z = np.random.normal(size=(num_images, self.z_size))
            return sess.run(self.x_mu, feed_dict={self.sample_z: sample_z})

    """" Implementation """
    def _build_graph(self):
        with tf.name_scope('placeholder'):
            self.inputs = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3), name='inputs')
            self.is_training = tf.placeholder(tf.bool, (None), name='is_training')
                
        self.normalized_images = self._preprocess_images(self.inputs)
        self.z_mu, self.z_logsigma = self._encode(self.normalized_images)
        self.sample_z = self._sample_norm(self.z_mu, self.z_logsigma, 'sample_z')
        self.x_mu = self._decode(self.sample_z, reuse=self.reuse)
        self.loss = self._loss(self.z_mu, self.z_logsigma, self.normalized_images, self.x_mu)

        # add image to tf.summary
        with tf.name_scope('image'):
            # record an original image
            tf.summary.image('original_image_', self.inputs[:1])
            # record a generated image
            timage = tf.cast((tf.clip_by_value(self.x_mu * self.std + self.mean, 0, 255)), tf.uint8)
            tf.summary.image('generated_image_', timage)

    def _preprocess_images(self, images):
        with tf.name_scope('preprocessing'):
            self.mean, var = tf.nn.moments(images, [0, 1, 2])
            self.std = tf.sqrt(var)
            normalized_images = (images - self.mean) / self.std
        return normalized_images

    def _encode(self, inputs):                                 
        x = inputs
        
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
        with tf.variable_scope('encoder', reuse=True):
            w = tf.get_variable('conv2d/kernel')
            tf.summary.histogram('conv0_weights_', w)

        return mu, logsigma

    def _sample_norm(self, mu, logsigma, scope):
        with tf.name_scope(scope):
            sigma = tf.exp(logsigma)
            epsilon = tf.random_normal(tf.shape(mu))

            sample = mu + sigma * epsilon
        tf.summary.histogram(scope + '_', sample)

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
            
        # record some weights
        with tf.variable_scope('decoder', reuse=True):
            w = tf.get_variable('conv2d_transpose_5/kernel')
            tf.summary.histogram('convtrans5_weights_', w)

        return x_mu

    def _loss(self, mu, logsigma, labels, predictions):
        with tf.name_scope('loss'):
            with tf.name_scope('kl_loss'):
                KL_loss = utils.kl_loss(mu, logsigma)
                beta = self._args[self._name]['beta']
                beta_KL = beta * KL_loss

            with tf.name_scope('reconstruction_loss'):
                reconstruction_loss = tf.losses.mean_squared_error(labels, predictions)
            
            with tf.name_scope('regularization'):
                l2_loss = tf.losses.get_regularization_loss(self._name)
            
            with tf.name_scope('total_loss'):
                loss = reconstruction_loss + beta_KL + l2_loss

            tf.summary.scalar('reconstruction_error_', reconstruction_loss)
            tf.summary.scalar('beta_', beta)
            tf.summary.scalar('kl_loss_', KL_loss)
            tf.summary.scalar('beta_kl_', beta_KL)
            tf.summary.scalar('l2_regularization_', l2_loss)
            tf.summary.scalar('total_loss_', loss)

        return loss

    def _optimize(self, loss):
        # params for optimizer
        init_learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer([0]), trainable=False)
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 1000, 0.95, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            tf.summary.scalar('learning_rate_', learning_rate)

        with tf.control_dependencies(update_ops):
            opt_op = self._optimizer.minimize(loss, global_step=global_step)

        return opt_op
