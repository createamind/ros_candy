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
            NO_SUCH_FILE = 'Missing_file'
            models, key = self._models_key() 
            path_prefix = models[key] if key in models else NO_SUCH_FILE
            if path_prefix != NO_SUCH_FILE:
                try:
                    self._saver.restore(sess, path_prefix)
                    print("Params for {} are restored.".format(self._name))
                    return 
                except:
                    del models[key]
            print('No saved model for "{}" is found. \nStart Training from Scratch!'.format(self._name))

    def save(self, sess):
        if self._saver:
            models, key = self._models_key()
            path_prefix = self._saver.save(sess, os.path.join(sys.path[0], 'saved_models/' + self._args['model_name'], str(self._name)))
            utils.save_args({key: path_prefix}, filename='models.yaml')

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

    def _models_key(self):
        return utils.load_args('models.yaml'), self._name + self._args['model_name']
