from modules.beta_vae import BetaVAE
from modules.utils.distribution import log_normal
import modules.utils.utils as utils
import numpy as np
import tensorflow as tf

""""
I comment out things corresponding to computing 'constant' in self._loss()
since 'constant' seems to contribute nothing to back-propagation 
"""

class TCVAE(BetaVAE):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        # self.dataset_size = args['dataset_size']
        self.alpha = args[name]['alpha']
        self.beta = args[name]['beta']
        self.gamma = args[name]['gamma']

        super(TCVAE, self).__init__(name, args, reuse)

    """ Implementation """
    def _loss(self, mu, logsigma, predictions, labels):
        with tf.name_scope('loss'):
            with tf.name_scope('kl_loss'):
                logpz = tf.reduce_sum(log_normal(self.sample_z, 0., 0.), axis=1)                       # log(p(z))
                logqz_condx = tf.reduce_sum(log_normal(self.sample_z, mu, logsigma), axis=1)           # log(q(z|x))

                # log(q|z)
                logqz_condx_expanded = log_normal(tf.expand_dims(self.sample_z, axis=1), 
                                                  tf.expand_dims(mu, axis=0), 
                                                  tf.expand_dims(logsigma, axis=0))

                # constant = np.log(self.batch_size * self.dataset_size)
                # logqz_marginal_product = tf.reduce_sum(utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False) - constant, axis=1)

                # sum(log(sum(q(zi|x))))
                logqz_marginal_product = tf.reduce_sum(utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False), axis=1)
                # log(sum(q(z|x)))
                logqz = utils.logsumexp(tf.reduce_sum(logqz_condx_expanded, axis=2), axis=1, keepdims=False)# - constant
                
                # divided by image_size**2 because we use MSE for reconstruction loss
                MI_loss = tf.reduce_mean(logqz_condx - logqz) / (self._args['image_size']**2)
                TC_loss = tf.reduce_mean(logqz - logqz_marginal_product) / (self._args['image_size']**2)
                dimension_wise_KL = tf.reduce_mean(logqz_marginal_product - logpz) / (self._args['image_size']**2)

                KL_loss = self.alpha * MI_loss + self.beta * TC_loss + self.gamma * dimension_wise_KL

            with tf.name_scope('reconstruction_error'):
                reconstruction_loss = tf.losses.mean_squared_error(labels, predictions)
            
            with tf.name_scope('l2_regularization'):
                l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.name_scope('total_loss'):
                loss = reconstruction_loss + KL_loss + l2_loss

            tf.summary.scalar('Reconstruction_error_', reconstruction_loss)
            tf.summary.scalar('MI_loss_', MI_loss)
            tf.summary.scalar('TC_loss_', TC_loss)
            tf.summary.scalar('Dimension_wise_KL_', dimension_wise_KL)
            tf.summary.scalar('KL_loss_', KL_loss)
            tf.summary.scalar('L2_loss_', l2_loss)
            tf.summary.scalar('Total_loss_', loss)

        return loss

