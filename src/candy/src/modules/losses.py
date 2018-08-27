from __future__ import print_function, division, absolute_import
import tensorflow as tf
from modules.module import Module

class MSELoss(Module):

	def __init__(self, predict, label, *args, **kwargs):
		self.predict = predict
		self.label = label
		super(MSELoss, self).__init__(*args, **kwargs)

    def _build_net(self, inputs, is_training, reuse):
		
		loss = tf.reduce_mean(tf.losses.mean_squared_error(self.label, self.predict))
		tf.summary.scalar(self.name + '_loss', loss)

		return loss


class VAELoss():

    def __init__(self, args, name, recon_x, x, mu, logsigma):
        self.args = args
        self.name = name
        self.recon_x = recon_x
        self.x = x
        self.mu = mu
        self.logsigma = logsigma

    def inference(self):
        const = 1 / (self.args['batch_size'] * self.args['x_dim'] * self.args['y_dim'])
        
        # self.x = tf.Print(self.x, [self.x])
        # self.recon_x = tf.Print(self.recon_x, [self.recon_x])

        self.recon = const * tf.reduce_sum(tf.squared_difference(self.x, self.recon_x))
        self.vae = const * -0.5 * tf.reduce_sum(1.0 + 2.0 * self.logsigma - tf.square(self.mu) - tf.exp(2 * self.logsigma))

        tf.summary.scalar(self.name + 'loss_vae', self.vae)
        tf.summary.scalar(self.name + 'loss_recon', self.recon)

        return tf.reduce_sum(self.recon + 250 * self.vae)


# class CrossEntropyLoss:

# 	def __init__(self, args, name, predict, label):
# 		self.args = args
# 		self.name = name
# 		self.predict = predict
# 		self.label = label

# 	def inference(self):
# 		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.predict))
# 		tf.summary.scalar(self.name + 'loss', loss)

# 		return loss