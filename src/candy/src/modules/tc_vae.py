from modules.beta_vae import BetaVAE
from modules.utils.distribution import log_normal
import modules.utils.utils as utils
from math import log

""""
I comment out things corresponding to computing 'constant' in self._loss()
since 'constant' seems to contribute nothing to back-propagation 
"""

class TCVAE(BetaVAE):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        # self.dataset_size = args['dataset_size']
        self.alpha = args[self._name]['alpha']
        self.beta = args[self._name]['beta']
        self.gamma = args[self._name]['gamma']

        super(TCVAE, self).__init__(self, name, args, reuse)

    """ Implementation """
    def _loss(self, mu, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                logpz = log_normal(self.sample_z, 0, 0)                     # log(p(z))
                logqz_condx = log_normal(self.sample_z, mu, logsigma)       # log(q(z|x))

                # log(q|z)
                logqz_condx_expanded = log_normal(tf.expand_dims(self.sample_z, axis=1), 
                                                  tf.expand_dims(mu, axis=0), 
                                                  tf.expand_dims(logsigma, axis=0))

                # constant = log(self.batch_size * self.dataset_size)
                # logqz_marginal_product = tf.reduce_sum(utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False) - constant, axis=1)
                # sum(log(sum(q(zi|x))))
                logqz_marginal_product = tf.reduce_sum(utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False), axis=1)
                # log(sum(q(z|x)))
                logqz = utils.logsumexp(tf.reduce_sum(logqz_condx_expanded, axis=2), axis=1, keepdims=False)# - constant

                MI_loss = tf.reduce_mean(logqz_condx - logqz)
                TC_loss = tf.reduce_mean(logqz - logqz_marginal_product)
                Dimension_wise_KL = tf.reduce_mean(logqz_marginal_product - logpz)

                KL_loss = self.alpha * MI_loss + self.beta * TC_loss + self.gamma * Dimension_wise_KL

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            with tf.variable_scope('l2_regularization', reuse=self.reuse):
                l2_loss = t f.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.variable_scope('total_loss', reuse=self.reuse):
                loss = reconstruction_loss + KL_loss + l2_loss

            tf.summary.scalar('Reconstruction_error', reconstruction_loss)
            tf.summary.scalar('MI_loss', MI_loss)
            tf.summary.scalar('TC_loss', TC_loss)
            tf.summary.scalar('Dimension_wise_KL', Dimension_wise_KL)
            tf.summary.scalar('KL_loss', KL_loss)
            tf.summary.scalar('L2_loss', l2_loss)
            tf.summary.scalar('Total_loss', loss)

        return loss

