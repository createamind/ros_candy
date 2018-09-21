from modules.beta_vae import BetaVAE
from modules.utils.distribution import log_normal
import modules.utils.utils as utils
import modules.utils.tf_utils as tf_utils
import numpy as np
import tensorflow as tf

class TCVAE(BetaVAE):
    """ Interface """
    def __init__(self, name, args, reuse=False, build_graph=True, log_tensorboard=False):
        self.dataset_size = args['dataset_size']
        self.batch_size = args['batch_size']
        self.alpha = args[name]['alpha']
        self.beta = args[name]['beta']
        self.gamma = args[name]['gamma']

        super(TCVAE, self).__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    """ Implementation """
    def _loss(self, mu, logsigma, predictions, labels):
        with tf.name_scope('loss'):
            with tf.name_scope('kl_loss'):
                kl_penalizing_size = self.z_size // 2
                sample_z_penalized = self.sample_z[:, : kl_penalizing_size]
                mu_penalized = mu[:, : kl_penalizing_size]
                logsigma_penalized = logsigma[:, : kl_penalizing_size]
                logpz = tf.reduce_sum(log_normal(sample_z_penalized, 0., 0.), axis=1)                                           # log(p(z))
                logqz_condx = tf.reduce_sum(log_normal(sample_z_penalized, mu_penalized, logsigma_penalized), axis=1)           # log(q(z|x))

                # log(q|z)
                logqz_condx_expanded = log_normal(tf.expand_dims(sample_z_penalized, axis=1), 
                                                  tf.expand_dims(mu_penalized, axis=0), 
                                                  tf.expand_dims(logsigma_penalized, axis=0))

                constant = np.log(self.batch_size * self.dataset_size)
                
                # sum(log(sum(q(zi|x))) - constant)
                logqz_marginal_product = tf.reduce_sum(tf_utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False) - constant, axis=1)

                # log(sum(q(z|x))) - constant
                logqz = tf_utils.logsumexp(tf.reduce_sum(logqz_condx_expanded, axis=2), axis=1, keepdims=False) - constant
                
                # divided by image_size**2 because we use MSE for reconstruction loss
                MI_loss = self.alpha * tf.reduce_mean(logqz_condx - logqz) / (self._args['image_size']**2)
                TC_loss = self.beta * tf.reduce_mean(logqz - logqz_marginal_product) / (self._args['image_size']**2)
                dimension_wise_KL = self.gamma * tf.reduce_mean(logqz_marginal_product - logpz) / (self._args['image_size']**2)

                KL_loss = MI_loss + TC_loss + dimension_wise_KL

            with tf.name_scope('reconstruction_error'):
                reconstruction_loss = tf.losses.mean_squared_error(labels, predictions)
            
            with tf.name_scope('l2_regularization'):
                l2_loss = tf.losses.get_regularization_loss(self.name, name='l2_regularization')
            
            with tf.name_scope('total_loss'):
                loss = reconstruction_loss + KL_loss + l2_loss

            if self.log_tensorboard:
                tf.summary.scalar('Reconstruction_error_', reconstruction_loss)
                tf.summary.scalar('MI_loss_', MI_loss)
                tf.summary.scalar('TC_loss_', TC_loss)
                tf.summary.scalar('Dimension_wise_KL_', dimension_wise_KL)
                tf.summary.scalar('KL_loss_', KL_loss)
                tf.summary.scalar('L2_loss_', l2_loss)
                tf.summary.scalar('Total_loss_', loss)

        return loss

