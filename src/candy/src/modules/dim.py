import math
import numpy as np
import tensorflow as tf
from modules.beta_vae import BetaVAE
import modules.utils.utils as utils
import modules.utils.tf_utils as tf_utils

""" 
Unfinished, waiting for incoporation
Only compute the local mutual information
The global one shouldn't be much different
"""
class DIM(BetaVAE):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        super(DIM, self).__init__(name, args, reuse)

    def _loss(self, mu, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('local_MI', reuse=self.reuse):
                T_joint, T_prod, E_joint, E_prod = self._score(self.z_mu)

                local_MI = E_joint - E_prod
            
            with tf.name_scope('reconstruction_error'):
                reconstruction_loss = tf.losses.mean_squared_error(labels, predictions)
            
            with tf.name_scope('l2_regularization'):
                l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')

            with tf.name_scope('total_loss'):
                loss = reconstruction_loss - 10 * local_MI + l2_loss

            tf.summary.scalar('T_joint_', tf.reduce_mean(T_joint))
            tf.summary.scalar('T_prod_', tf.reduce_mean(T_prod))
            tf.summary.scalar('E_joint_', E_joint)
            tf.summary.scalar('E_prod_', E_prod)
            tf.summary.scalar('Reconstruction_error_', reconstruction_loss)
            tf.summary.scalar('Local_MI_', local_MI)
            tf.summary.scalar('L2_loss_', l2_loss)
            tf.summary.scalar('Total_loss_', loss)
        
        return loss

    """ Implementation """
    def _score(self, z):
        with tf.name_scope('discriminator'):
            T_joint = self._get_score(z)
            T_prod = self._get_score(z, shuffle=True)

            log2 = math.log(2.)
            E_joint = log2 - tf.reduce_mean(tf.math.softplus(-T_joint))
            E_prod = tf.reduce_mean(tf.math.softplus(-T_prod) + T_prod) - log2

        return T_joint, T_prod, E_joint, E_prod

    def _get_score(self, z, shuffle=False):
        with tf.name_scope('score'):
            feature_map = self.dim_feature_map
            height, width, channels = feature_map.shape.as_list()[1:]
            z_channels = z.shape.as_list()[-1]
            
            # here's where the error is introduced
            if shuffle:
                original_map = feature_map
                feature_map = tf.random_shuffle(feature_map)
                feature_map = tf.stop_gradient(feature_map)
            
            # expand z
            z_padding = tf.tile(z, [1, height * width])
            z_padding = tf.reshape(z_padding, [-1, height, width, z_channels])

            feature_map = tf.concat([feature_map, z_padding], axis=-1)
            scores = self._local_discriminator(feature_map, shuffle)
            scores = tf.reshape(scores, [-1, height * width])

        return scores

    def _local_discriminator(self, feature_map, reuse):
        with tf.variable_scope('local_discriminator', reuse=reuse):
            x = self._conv(feature_map, 256, 1, kernel_initializer=tf_utils.kaiming_initializer())
            x = tf.nn.relu(x)
            x = self._conv(feature_map, 128, 1, kernel_initializer=tf_utils.kaiming_initializer())
            x = tf.nn.relu(x)
            x = self._conv(feature_map, 1, 1)

        return x

    # def _random_permute(self, x):
    #     _, d2, d3 = x.shape.as_list()
    #     d1 = self._args['batch_size']
    #     b = np.random.rand(d1, d2)
    #     idx = np.argsort(b, 0)
    #     adx = np.arange(0, d2)
    #     adx = np.broadcast_to(adx, (d1, d2))
    #     x = x[idx, adx]

    #     return x
