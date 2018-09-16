import tensorflow as tf
import modules.utils.utils as utils
import os
import sys

class Module(object):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        self._args = args
        self._name = name
        self.reuse = reuse

        with tf.variable_scope(self._name, reuse=self.reuse):
            self.l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])

            self._build_graph()

            self.opt_op = self._optimize(self.loss)

            collection = tf.global_variables(self._name)
            if len(collection) > 0:
                self._saver = tf.train.Saver(collection)
            else:
                self._saver = None
    
    def restore(self, sess):
        if self._saver:
            key = self._name + '_path_prefix'
            no_such_file = 'Missing_file'
            models = load_args('models.yaml')
            path_prefix = models[key] if key in models else no_such_file
            if path_prefix != no_such_file:
                try:
                    self._saver.restore(sess, path_prefix)
                    print("Params for {} are restored".format(self._name))
                except:
                    del self._args[key]
    
    def save(self, sess):
        if self._saver:
            path_prefix = self._saver.save(sess, os.path.join(sys.path[0], 'models/trial/0.0001-0.95', str(self._name)))
            key = self._name + '_path_prefix'
            self._args[key] = path_prefix
            utils.save_args({key: path_prefix}, self._args, 'models.yaml')

    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError

    def _optimize(self, loss):
        raise NotImplementedError

    def _dense(self, x, units, kernel_initializer=utils.xavier_initializer()):
        return tf.layers._dense(x, units, kernel_initializer=kernel_initializer, 
                               kernel_regularizer=self.l2_regularizer)

    def _dense_bn_relu(self, x, units, kernel_initializer=utils.kaiming_initializer()):
        x = self._dense(x, units, kernel_initializer=kernel_initializer)
        x = utils.bn_relu(x, self.is_training)

        return x

    def _conv(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.xavier_initializer()): 
        return tf.layers.conv2d(x, filters, filter_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=self.l2_regularizer)

    def _conv_bn_relu(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.kaiming_initializer()):
        x = self._conv(x, filters, filter_size, strides, padding=padding, kernel_initializer=kernel_initializer)
        x = utils.bn_relu(x, self.is_training)

        return x
    
    def _convtrans(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.xavier_initializer()): 
        return tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding=padding, 
                                          kernel_initializer=kernel_initializer, kernel_regularizer=self.l2_regularizer)
    
    def _convtrans_bn_relu(self, x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.kaiming_initializer()):
        x = self._convtrans(x, filters, filter_size, strides, padding=padding)
        x = utils.bn_relu(x, self.is_training)

        return x