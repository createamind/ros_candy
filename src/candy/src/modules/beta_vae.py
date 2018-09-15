import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import modules.utils.utils as utils
from modules.module import Module

class BetaVAE(Module):
    def __init__(self, name, args, reuse=False, z_size=64):
        self.z_size = z_size
        self.image_size = args['image_size']
        super(BetaVAE, self).__init__(name, args, reuse)

        self.epsilon = 1e-8        
        
        with tf.variable_scope(self._name, reuse=True):
            # add image summaries at training time
            with tf.variable_scope('image', reuse=self.reuse):
                # record an original image
                timage = tf.cast((tf.clip_by_value(self.inputs, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('original_image', timage[:1])
                # record a generated image
                timage = tf.cast((tf.clip_by_value(self.x_mu, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('generated_image', timage[:1])

    def generate(self, sess, num_images=1):
        with tf.variable_scope(self._name, reuse=True):
            sample_z = np.random.normal(size=(num_images, self.z_size))
            return sess.run(self.x_mu, feed_dict={self.sample_z: sample_z})

    def _build_graph(self):
        with tf.variable_scope(self._name, reuse=True):
            with tf.variable_scope('placeholder', reuse=self.reuse):
                self.inputs = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3), name='inputs')
                self.is_training = tf.placeholder(tf.bool, (None), name='is_training')
                
        self.z_mu, self.z_logsigma = self._encode()
        self.sample_z = self._sample_norm(self.z_mu, self.z_logsigma, 'sample_z')
        self.x_mu = self._decode(self.sample_z, reuse=self.reuse)
        self.loss = self._loss(self.z_mu, self.z_logsigma, self.x_mu, self.inputs)

    def _encode(self):                                 
        x = self.inputs
        
        # encoder net
        with tf.variable_scope('encoder', reuse=self.reuse):
            x = self.conv_bn_relu(x, 16, 4, 2)           # x = 160, 160, 16
            x = self.conv_bn_relu(x, 32, 4, 2)           # x = 80, 80, 32
            x = self.conv_bn_relu(x, 64, 4, 2)           # x = 40, 40, 64
            x = self.conv_bn_relu(x, 128, 4, 2)          # x = 20, 20, 128
            x = self.conv_bn_relu(x, 256, 4, 2)          # x = 10, 10, 256
            x = self.conv_bn_relu(x, 512, 4, 2)          # x = 5, 5, 512

            """ Version without dense layer
            x = self.conv(x, 2 * self.z_size, 5, padding='valid', kernel_initializer=utils.xavier_initializer())  
            # x = 1, 1, 2 * z_size

            x = tf.reshape(x, [-1, 2 * self.z_size])
            """
            """ Version with dense layers """
            x = tf.reshape(-1, [-1, 5 * 5 * 512])
            x = self.dense_bn_relu(x, 512)
            x = self.dense(x, 512)
            
            mu, logsigma = tf.split(x, 2, -1)

            # record some weights
        if not self.reuse:
            with tf.variable_scope('encoder', reuse=True):
                w = tf.get_variable('conv2d/kernel')
                tf.summary.histogram('conv0_weights', w)

        return mu, logsigma

    def _sample_norm(self, mu, logsigma, scope, reuse=None):
        with tf.variable_scope(scope, reuse=self.reuse if reuse is None else reuse):
            std = tf.exp(logsigma)
            epsilon = tf.random_normal(tf.shape(mu))

            sample_z = mu + std * epsilon
            tf.summary.histogram(scope, sample_z)

        return sample_z

    def _decode(self, sample_z, reuse):
        x = sample_z

        # decoder net
        with tf.variable_scope('decoder', reuse=reuse):
            """ Version without dense layer
            x = tf.reshape(x, [-1, 1, 1, self.z_size])                  # x = 1, 1, z_size

            x = self.convtrans_bn_relu(x, 512, 5, 1, padding='valid')   # x = 5, 5, 512
            """
            """ Version with dense layers """
            x = self.dense_bn_relu(x, 512)
            x = self.dense_bn_relu(x, 5 * 5 * 512)
            x = tf.reshape(x, [-1, 5, 5, 512])                          # x = 5, 5, 512

            x = self.convtrans_bn_relu(x, 256, 4, 2)                    # x = 10, 10, 256
            x = self.convtrans_bn_relu(x, 128, 4, 2)                    # x = 20, 20, 128
            x = self.convtrans_bn_relu(x, 64, 4, 2)                     # x = 40, 40, 64
            x = self.convtrans_bn_relu(x, 32, 4, 2)                     # x = 80, 80, 32
            x = self.convtrans_bn_relu(x, 16, 4, 2)                     # x = 160, 160, 16
            x = self.conv_transpose(x, 3, 4, 2, kernel_initializer=utils.xavier_initializer())
            # x = 320, 320, 3

            x_mu = x
            
        if not self.reuse:
            with tf.variable_scope('decoder', reuse=True):
                # record some weights
                w = tf.get_variable('conv2d_transpose_5/kernel')
                tf.summary.histogram('conv_trans5_weights', w)

        return x_mu

    def _loss(self, mu, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                KL_loss = utils.kl_loss(mu, logsigma)
                MAX_BETA = 1
                beta = tf.get_variable('beta', shape=(), initializer=tf.constant_initializer([0.01]), trainable=False, dtype=tf.float32)
                new_beta = tf.assign(beta, tf.minimum(1.01 * beta, MAX_BETA))
                beta_KL = beta * KL_loss / (160**2)

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.control_dependencies([new_beta]):
                loss = reconstruction_loss + beta_KL# + l2_loss

            tf.summary.scalar('reconstruction_error', reconstruction_loss)
            tf.summary.scalar('beta', beta)
            tf.summary.scalar('KL_loss', KL_loss)
            tf.summary.scalar('beta_KL', beta_KL)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total_loss', loss)

            return loss

    def _optimize(self, loss):
        # grad_clip = self._args[self._name]['grad_clip'] if 'grad_clip' in self._args[self._name] else 10
        
        with tf.variable_scope('optimizer', reuse=self.reuse):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # grad_var_pairs = self._optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(self._name))
            # if not self.reuse:
            #     tf.summary.histogram('gradient', grad_var_pairs[0][0])
            # grad_var_pairs = [(tf.clip_by_norm(grad, grad_clip), var) for grad, var in grad_var_pairs]

        with tf.control_dependencies(update_ops):
            # opt_op = self._optimizer.apply_gradients(grad_var_pairs)
            opt_op = self._optimizer.minimize(loss)

        return opt_op
