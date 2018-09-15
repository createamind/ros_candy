import math
import tensorflow as tf
from modules.beta_vae import BetaVAE
import modules.utils.utils as utils

""" 
Unfinished, waiting for incoporation
Only compute the local mutual information
The global one shouldn't be much different
"""
class DIM(BetaVAE):
    def __init__(self, name, args, reuse=False):
        super(DIM, self).__init__(name, args, reuse)

    def _loss(self, mu, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('local_MI', reuse=self.reuse):
                E_joint, E_prod = self._score(self.z_mu)

                local_MI = E_joint - E_prod

    def _score(self, z):
        T_joint = self._get_score(z)
        T_prod = self._get_score(z, shuffle=True)

        log2 = math.log(2.)
        E_joint = log2 - tf.math.softplus(-T_joint)
        E_prod = tf.math.softplus(-T_prod) + T_prod - log2

        return E_joint, E_prod

    def _get_score(self, z, shuffle=False):
        feature_map = self.dim_feature_map
        height, width, channels = feature_map.shape.as_list()[1:]

        if shuffle:
            feature_map = tf.random_shuffle(feature_map) # TODO: leave to improve
        
        z_padding = tf.reshape(tf.tile(z, [1, height * width]), [-1, height, width, channels])

        feature_map = tf.concat([feature_map, z_padding], axis=-1)
        scores = self._local_discriminator(feature_map)
        scores = tf.reshape(scores, [-1, height * width])

        return scores

    def _local_discriminator(self, feature_map):
        x = self._conv(feature_map, 512, 1, kernel_initializer=utils.kaiming_initializer())
        x = tf.nn.relu(x)
        x = self._conv(feature_map, 512, 1, kernel_initializer=utils.kaiming_initializer())
        x = tf.nn.relu(x)
        x = self._conv(feature_map, 1, 1)

        return x
