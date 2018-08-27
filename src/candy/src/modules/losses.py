from __future__ import print_function, division, absolute_import
import tensorflow as tf
from modules.module import Module

class MSELoss(Module):

    def __init__(self, predict, label, *args, **kwargs):
        self._predict = predict
        self._label = label
        super(MSELoss, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        
        loss = tf.reduce_mean(tf.losses.mean_squared_error(self._label, self._predict))
        if reuse == False: # Indicating that is_test = False
            tf.summary.scalar(self._name, loss)

        return loss


class VAELoss(Module):

    def __init__(self, mu, logsigma, *args, **kwargs):
        self._mu = mu
        self._logsigma = logsigma
        super(VAELoss, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        const = 1 / (self._args['batch_size'] * self._args['x_dim'] * self._args['y_dim'])
        loss = const * -0.5 * tf.reduce_sum(1.0 + 2.0 * self._logsigma - tf.square(self._mu) - tf.exp(2 * self._logsigma))
        if reuse == False: # Indicating that is_test = False
            tf.summary.scalar(self._name, loss)
        return loss


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