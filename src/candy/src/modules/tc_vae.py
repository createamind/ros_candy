from modules.beta_vae import BetaVAE
from modules.utils.distribution import log_normal
import modules.utils.utils as utils
from math import log


class TCVAE(BetaVAE):
    def __init__(self, name, args, reuse=False, z_size=64):
        self.dataset_size = args['dataset_size']
        self.alpha = args[self._name]['alpha']
        self.beta = args[self._name]['beta']
        self.gamma = args[self._name]['gamma']

        super(TCVAE, self).__init__(self, name, args, reuse, z_size)

    def _loss(self, mu, logsigma, predictions, labels, sample_z):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                logpz = log_normal(sample_z, mu, logsigma)
                logqz_condx = log_normal(sample_z, mu, logsigma)

                print(tf.expand_dims(sample_z, axis=1))
                print(tf.expand_dims(logsigma, 0).shape.as_list())
                logqz_condx_expanded = log_normal(tf.expand_dims(sample_z, axis=1), 
                                                  tf.expand_dims(mu, axis=0), 
                                                  tf.expand_dims(logsigma, axis=0))
                print(logqz_condx_expanded.shape.as_list())

                constant = log(self.batch_size * self.dataset_size)
                logqz_marginal_product = tf.reduce_sum(utils.logsumexp(logqz_condx_expanded, axis=1, keepdims=False) \
                                                                 - constant, axis=1)
                logqz = utils.logsumexp(tf.reduce_sum(logqz_condx_expanded, axis=2), axis=1, keepdims=False) - constant

                MI_loss = logqz_condx - logqz
                TC_loss = logqz - logqz_marginal_product
                Dimension_wise_KL = logqz_marginal_product - logpz

                KL_loss = self.alpha * MI_loss + self.beta * TC_loss + self.gamma * Dimension_wise_KL

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            l2_loss = t f.losses.get_regularization_loss(self._name, name='l2_loss')
            
            loss = reconstruction_loss + KL_loss# + l2_loss

            tf.summary.scalar('reconstruction_error', reconstruction_loss)
            tf.summary.scalar('MI_loss', MI_loss)
            tf.summary.scalar('TC_loss', TC_loss)
            tf.summary.scalar('Dimension_wise_KL', Dimension_wise_KL)
            tf.summary.scalar('KL_loss', KL_loss)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total_loss', loss)

            return loss

