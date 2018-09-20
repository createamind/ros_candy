import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import modules.utils.utils as utils
import modules.utils.tf_utils as tf_utils
import modules.utils.losses as losses
from modules.module import Module

class BetaVAE(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False, standard_normalization=True):
        self.z_size = args['z_size']
        self.image_size = args['image_size']
        self.beta = args[name]['beta']
        self.standard_normalization = standard_normalization
        super(BetaVAE, self).__init__(name, args, reuse)

    def generate(self, sess, num_images=1):
        with tf.name_scope(self._name):
            sample_z = np.random.normal(size=(num_images, self.z_size))
            generated_image = self._restore_images(self.x_mu)
            
            outputs = sess.run(generated_image, feed_dict={self.sample_z: sample_z})

            return outputs
    
    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self._name)
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

        # add images to tf.summary
        with tf.name_scope('image'):
            # record an original image
            tf.summary.image('original_image_', self.inputs[:1])
            tf.summary.histogram('original_image_hist_', self.inputs[:1])
            # timage = self._restore_images(self.normalized_images)
            # tf.summary.image('restored_image', timage[:1])
            # record a generated image
            timage = self._restore_images(self.x_mu)
            tf.summary.image('generated_image_', timage[:1])
            tf.summary.histogram('generated_image_hist_', timage[:1])

    def _preprocess_images(self, images):
        with tf.name_scope('preprocessing'):
            if self.standard_normalization:
                normalized_images, self.mean, self.std = tf_utils.standard_normalization(images)
            else:
                normalized_images = tf_utils.range_normalization(images)

        return normalized_images

    def _restore_images(self, normalized_images):
        with tf.name_scope('restore_image'):
            if self.standard_normalization:
                images = tf.cast(tf.clip_by_value(normalized_images * self.std + self.mean, 0, 255), tf.uint8)
            else:
                images = tf_utils.range_normalization(normalized_images, normalizing=False)

        return images

    def _encode(self, inputs):                                 
        x = inputs
        
        # encoder net
        with tf.variable_scope('encoder', reuse=self.reuse):
            x = self._conv_bn_relu(x, 16, 3, 2)                       # x = 160, 160, 16
            x = self._conv_pool_bn_relu(x, 16, 3, 2)                  # x = 80, 80, 32
            x = self._conv_pool_bn_relu(x, 32, 3, 2)                  # x = 40, 40, 64
            x = self._conv_pool_bn_relu(x, 64, 3, 2)                  # x = 20, 20, 128
            x = self._conv_pool_bn_relu(x, 128, 3, 2)                 # x = 10, 10, 256
            x = self._conv_pool_bn_relu(x, 256, 3, 2)                 # x = 5, 5, 512
            self.dim_feature_map = x
            """ Version without dense layer """
            x = self._conv(x, 2 * self.z_size, 5, padding='valid', kernel_initializer=tf_utils.xavier_initializer())

            x = tf.reshape(x, [-1, 2 * self.z_size])

            mu, logsigma = tf.split(x, 2, -1)

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
            x = self._convtrans_bn_relu(x, 256, 3, 2)                    # x = 10, 10, 256
            x = self._convtrans_bn_relu(x, 128, 3, 2)                    # x = 20, 20, 128
            x = self._convtrans_bn_relu(x, 64, 3, 2)                     # x = 40, 40, 64
            x = self._convtrans_bn_relu(x, 32, 3, 2)                     # x = 80, 80, 32
            x = self._convtrans_bn_relu(x, 16, 3, 2)                     # x = 160, 160, 16
            x = self._convtrans(x, 3, 3, 2, kernel_initializer=tf_utils.xavier_initializer())
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
                KL_loss = self.beta * losses.kl_loss(mu, logsigma) / (self._args['image_size']**2)    # divided by image_size**2 because we use MSE for reconstruction loss

            with tf.name_scope('reconstruction_loss'):
                reconstruction_loss = tf.losses.mean_squared_error(labels, predictions)

            with tf.name_scope('regularization'):
                l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_regularization')
            
            with tf.name_scope('total_loss'):
                loss = reconstruction_loss + KL_loss + l2_loss

            tf.summary.scalar('Reconstruction_error_', reconstruction_loss)
            tf.summary.scalar('KL_loss_', KL_loss)
            tf.summary.scalar('L2_loss_', l2_loss)
            tf.summary.scalar('Total_loss_', loss)
        
        return loss

    def _optimize(self, loss):
        # params for optimizer
        init_learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999
        decay_rate = self._args[self._name]['decay_rate'] if 'decay_rate' in self._args[self._name] else 0.95
        decay_steps = self._args[self._name]['decay_steps'] if 'decay_steps' in self._args[self._name] else 1000

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer([0]), trainable=False)
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            tf.summary.scalar('learning_rate_', learning_rate)

        with tf.control_dependencies(update_ops):
            opt_op = self._optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('gradients'):
            grad_var_pairs = self._optimizer.compute_gradients(loss)
            for grad, var in grad_var_pairs:
                if grad is None:
                    continue
                tf.summary.histogram(var.name.replace(':0', '/gradient'), grad)

        with tf.name_scope('weights'):
            for var in self.trainable_variables:
                tf.summary.histogram(var.name.replace(':0', ''), var)
            

        return opt_op
